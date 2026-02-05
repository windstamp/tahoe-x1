#!/usr/bin/env python3
"""
Analyze profiling results from Tahoe-x1 training.

Usage:
    python analyze_tahoe_profile.py --profile_dir ./profiling
    python analyze_tahoe_profile.py --profile_dir ./profiling --show_all
    python analyze_tahoe_profile.py --profile_dir ./profiling --model_checkpoint path/to/checkpoint.pt
"""

import argparse
import os
from profile_utils import analyze_layer_shapes, analyze_unique_operators, analyze_chrome_trace, extract_operator_info, analyze_model_summary


def main():
    parser = argparse.ArgumentParser(description='Analyze Tahoe-x1 model profiling results')
    parser.add_argument('--profile_dir', type=str, default='./profiling',
                      help='Directory containing profiling results')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                      help='Path to model checkpoint for architecture analysis')
    parser.add_argument('--show_all', action='store_true',
                      help='Show all operators instead of just top 30')
    args = parser.parse_args()
    
    layer_shapes_file = os.path.join(args.profile_dir, 'layer_shapes.json')
    trace_file = os.path.join(args.profile_dir, 'trace.json')
    
    print(f"\n{'='*80}")
    print(f"Analyzing Tahoe-x1 Model Profiling Results")
    print(f"{'='*80}\n")
    
    # Load and analyze model if checkpoint provided
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        print(f"Loading model from: {args.model_checkpoint}")
        try:
            import torch
            
            checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
            # Try to extract model config from checkpoint
            if 'hyper_parameters' in checkpoint:
                print("\nModel hyperparameters found in checkpoint")
                analyze_model_summary_from_checkpoint(checkpoint)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Analyze layer shapes
    if os.path.exists(layer_shapes_file):
        analyze_layer_shapes(layer_shapes_file)
        analyze_unique_operators(layer_shapes_file)
    else:
        print(f"Layer shapes file not found: {layer_shapes_file}")
    
    # Analyze Chrome trace
    if os.path.exists(trace_file):
        sorted_ops = analyze_chrome_trace(trace_file)
        
        # Extract detailed operator info
        operators_file = os.path.join(args.profile_dir, 'operators_detail.json')
        extract_operator_info(trace_file, operators_file)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Summary")
        print(f"{'='*80}\n")
        print(f"Total unique operators: {len(sorted_ops)}")
        if sorted_ops:
            print(f"Most expensive operator: {sorted_ops[0][0]} ({sorted_ops[0][1]['total_time']:.2f} us)")
            print(f"\nTop 10 operators by total time:")
            for i, (name, stats) in enumerate(sorted_ops[:10], 1):
                print(f"  {i}. {name}: {stats['total_time']:.2f} us ({stats['count']} calls)")
        print(f"\nFull results saved to: {args.profile_dir}")
    else:
        print(f"Trace file not found: {trace_file}")


def analyze_model_summary_from_checkpoint(checkpoint):
    """Analyze model summary from checkpoint hyperparameters."""
    print(f"\n{'='*80}")
    print(f"Model Configuration from Checkpoint")
    print(f"{'='*80}\n")
    
    hparams = checkpoint.get('hyper_parameters', {})
    
    important_keys = [
        'hidden_dim', 'n_encoder_layers', 'n_decoder_layers',
        'transformer_backbone_kwargs', 'input_dim', 'gene_dim', 'output_dim',
        'pert_dim', 'batch_dim', 'dropout', 'learning_rate'
    ]
    
    for key in important_keys:
        if key in hparams:
            value = hparams[key]
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    # Calculate approximate model size
    if 'state_dict' in checkpoint:
        total_params = sum(p.numel() for p in checkpoint['state_dict'].values())
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Estimated model size: {total_params * 4 / 1024**3:.2f} GB (float32)")


if __name__ == '__main__':
    main()
