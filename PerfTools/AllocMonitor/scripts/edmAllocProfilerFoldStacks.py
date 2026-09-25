#!/usr/bin/env python3

import argparse
import re
import sys

def parse_quantity_value_line(line):
    """Parse a line containing quantity-value pairs.
    Returns an ordered list of (quantity, value) tuples."""
    tokens = line.split()
    pairs = []
    i = 0
    while i + 1 < len(tokens):
        try:
            value = int(tokens[i + 1])
            pairs.append((tokens[i], value))
            i += 2
        except ValueError:
            i += 1
    return pairs

def parse_stack_frame_function(line):
    """Extract the function name from a stack frame line."""
    stripped = line.lstrip()
    # Remove frame number prefix (digits followed by '#' and optional whitespace)
    rest = re.sub(r'^\d+#\s*', '', stripped)
    # Split on the last ' at ' to separate function name from file:line
    parts = rest.rsplit(' at ', 1)
    return parts[0]

def is_stack_frame_line(line):
    """Return True if the line is a stack frame line."""
    return bool(re.match(r'\s*\d+#', line))

def main(args):
    quantity = args.quantity
    first_quantity_name = None  # first quantity name seen across all records
    error_occurred = False
    current_pairs = None
    current_frames = []

    def emit_record(pairs, frames):
        nonlocal quantity, first_quantity_name, error_occurred

        if not pairs:
            return

        # Validate consistent first quantity across records
        record_first = pairs[0][0]
        if first_quantity_name is None:
            first_quantity_name = record_first
            if quantity is None:
                quantity = record_first
        elif record_first != first_quantity_name:
            print(f'Error: inconsistent first quantity: expected {first_quantity_name!r}, got {record_first!r}',
                  file=sys.stderr)
            error_occurred = True
            return

        qty_dict = dict(pairs)
        if quantity not in qty_dict:
            print(f'Error: quantity {quantity!r} not found in record', file=sys.stderr)
            error_occurred = True
            return

        value = qty_dict[quantity]
        reversed_frames = list(reversed(frames))
        print('; '.join(reversed_frames) + ' ' + str(value))

    with open(args.input) as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()

            if not stripped or stripped.startswith('#'):
                continue

            if is_stack_frame_line(line):
                if current_pairs is not None:
                    current_frames.append(parse_stack_frame_function(line))
            else:
                # New quantity-value line: emit the previous record first
                if current_pairs is not None:
                    emit_record(current_pairs, current_frames)
                    if error_occurred:
                        sys.exit(1)
                current_pairs = parse_quantity_value_line(stripped)
                current_frames = []

    # Emit the final record
    if current_pairs is not None:
        emit_record(current_pairs, current_frames)
        if error_occurred:
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform output strack trace files from IntrusiveAllocProfiler to format usable e.g. for flame graphs'
    )
    parser.add_argument('input', help='Path to the input file to analyze')
    parser.add_argument('-q', '--quantity', default=None,
                        help='Name of the quantity to print (default: first quantity in file)')
    args = parser.parse_args()

    main(args)
