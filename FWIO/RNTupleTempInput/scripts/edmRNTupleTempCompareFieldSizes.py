#!/usr/bin/env python3

import argparse
import subprocess
import re
import sys

def parse_size(size_str):
    """Parse size string like '4820 B' or '1.5 KB' into bytes."""
    size_str = size_str.strip()
    match = re.match(r'([\d.]+)\s*([A-Z]*B?)', size_str)
    if not match:
        raise ValueError(f"Cannot parse size: {size_str}")
    
    value = float(match.group(1))
    unit = match.group(2).upper()
    
    if unit == 'B' or unit == '':
        return int(value)
    elif unit == 'KB':
        return int(value * 1024)
    elif unit == 'MB':
        return int(value * 1024 * 1024)
    elif unit == 'GB':
        return int(value * 1024 * 1024 * 1024)
    else:
        raise ValueError(f"Unknown unit: {unit}")


def run_edmRNTupleTempStorage(filename):
    try:
        result = subprocess.run(
            ['edmRNTupleTempStorage', '-p', filename],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running edmRNTupleTempStorage on {filename}:")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: edmRNTupleTempStorage command not found")
        sys.exit(1)


def parse_output(output):
    """Parse edmRNTupleTempStorage output into a dictionary of field name to size."""
    field_sizes = {}
    lines = output.split('\n')
    
    current_field = None
    for line in lines:
        # Look for field name line (starts with whitespace, then field name)
        field_match = re.match(r'\s+([^\s]+)\s+\[#\d+\]', line)
        if field_match:
            current_field = field_match.group(1)
            continue
        
        # Look for "Size on storage:" line
        if current_field and 'Size on storage:' in line:
            size_match = re.search(r'Size on storage:\s+(.+)', line)
            if size_match:
                size_str = size_match.group(1)
                field_sizes[current_field] = parse_size(size_str)
                current_field = None
    
    return field_sizes


def main(args):
    # Run edmRNTupleTempStorage on both files
    print(f"Processing {args.file_a}...")
    output_a = run_edmRNTupleTempStorage(args.file_a)
    
    print(f"Processing {args.file_b}...")
    output_b = run_edmRNTupleTempStorage(args.file_b)
    
    # Parse outputs
    sizes_a = parse_output(output_a)
    sizes_b = parse_output(output_b)
    
    # Check for fields present in only one file
    fields_only_in_a = set(sizes_a.keys()) - set(sizes_b.keys())
    fields_only_in_b = set(sizes_b.keys()) - set(sizes_a.keys())
    
    if fields_only_in_a or fields_only_in_b:
        error_msg = "Field mismatch between files:\n"
        if fields_only_in_a:
            error_msg += f"  Only in {args.file_a}: {', '.join(sorted(fields_only_in_a))}\n"
        if fields_only_in_b:
            error_msg += f"  Only in {args.file_b}: {', '.join(sorted(fields_only_in_b))}\n"
        raise ValueError(error_msg)
    
    # Compute size differences (A - B), keep only negative differences (size decreased)
    size_diffs = {}
    nfields_same = 0
    nfields_worse = 0
    for field in sizes_a.keys():
        diff = sizes_a[field] - sizes_b[field]
        if diff > 0:
            size_diffs[field] = diff
        elif diff == 0:
            nfields_same += 1
        else:
            nfields_worse += 1
    
    if not size_diffs:
        print("No fields with decreased size found.")
        return
    
    # Sort by size difference (largest first)
    sorted_fields = sorted(size_diffs.items(), key=lambda x: -x[1])
    
    # Calculate total size difference
    total_diff = sum(size_diffs.values())
    threshold = total_diff * (args.threshold/100.)
    
    # Accumulate from smallest until reaching threshold
    # Start from the end of the sorted list and work backwards
    accumulated = 0
    cutoff_index = len(sorted_fields)
    for i in range(len(sorted_fields) - 1, -1, -1):
        field, diff = sorted_fields[i]
        accumulated += diff
        if accumulated < threshold:
            cutoff_index = i
        else:
            break
    
    # Keep only fields before the cutoff (most negative ones)
    remaining_fields = sorted_fields[:cutoff_index]
    
    if not remaining_fields:
        print(f"No significant fields remaining after applying {args.threshold} % threshold.")
        return

    size_a = sum(sizes_a.values())
    max_reduction = total_diff/size_a * 100
    suggested_reduction = sum(x[1] for x in remaining_fields)
    
    # Print results
    print(f"\nSize of first file: {size_a:,} bytes")
    print(f"Max size reduction: {total_diff:,} bytes, {max_reduction} %")
    print(f"{args.threshold} % threshold: {threshold:,.2f} bytes")
    print(f"Suggested size reduction {suggested_reduction:,} bytes, {suggested_reduction/size_a*100}")
    print(f"Total number of fields: {len(sizes_a)}")
    print(f"Total number of increased fields: {nfields_worse}")
    print(f"Total number of equal     fields: {nfields_same}")
    print(f"Total number of decreased fields: {len(size_diffs)}")
    print(f"Fields accounting for largest {100-args.threshold} % decrease: {len(remaining_fields)}")
    
    print("\nFields with most significant size reductions (largest reduction first):")
    print("-" * 70)
    for field, diff in remaining_fields:
        print(f"{field:100s} {diff:10,} bytes")

    if args.asList:
        print("\n" + "=" * 70)
        print("\npythonList = [")
        for field, diff in remaining_fields:
            print(f"    '{field}',")
        print("]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare RNTuple storage sizes between two files and identify fields with reduced size.'
    )
    parser.add_argument('file_a', help='First RNTuple file (baseline)')
    parser.add_argument('file_b', help='Second RNTuple file (comparison)')
    parser.add_argument('--threshold', default=5, type=int, help='Leave out fields that have the smallest reduction up to THRESHOLD %% of the maximum size reduction from file_a to file_B. Default: 5 %%.')
    parser.add_argument('--asList', action='store_true', help='Print the field names as a python list')
    
    args = parser.parse_args()
    
    main(args)
