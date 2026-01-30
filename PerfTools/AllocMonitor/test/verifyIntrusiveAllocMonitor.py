#!/usr/bin/env python3

import sys
import re

def parse_message(line):
    """Parse a IntrusiveAllocMonitor message line and extract field values."""
    # Pattern: "Name: requested X added Y max alloc Z peak W nAlloc N nDealloc M"
    # Note: added can be negative
    pattern = r'^(?P<name>.+):\s+requested\s+(?P<requested>\d+)\s+added\s+(?P<added>-?\d+)\s+max alloc\s+(?P<max_alloc>\d+)\s+peak\s+(?P<peak>\d+)\s+nAlloc\s+(?P<nAlloc>\d+)\s+nDealloc\s+(?P<nDealloc>\d+)$'
    match = re.match(pattern, line.strip())
    if not match:
        raise ValueError(f"Malformed message line: {line}")

    data = match.groupdict()
    # Convert numeric fields to integers
    for key in ['requested', 'added', 'max_alloc', 'peak', 'nAlloc', 'nDealloc']:
        data[key] = int(data[key])
    return data

def read_messages():
    """Read stdin and extract IntrusiveAllocMonitor messages."""
    messages = []
    lines_iter = iter(sys.stdin)

    for line in lines_iter:
        if line.strip() == '%MSG-s IntrusiveAllocMonitor:':
            # Next line should be the message
            try:
                message_line = next(lines_iter)
            except StopIteration:
                raise ValueError("Unexpected end of input after %MSG-s line")

            # Next line should be %MSG
            try:
                end_marker = next(lines_iter)
            except StopIteration:
                raise ValueError("Unexpected end of input, expected %MSG")

            if end_marker.strip() != '%MSG':
                raise ValueError("Expected %MSG after message line")

            msg = parse_message(message_line)
            messages.append(msg)

    return messages

def verify_vector_fill(msg):
    """Verify conditions for 'Vector fill' message."""
    name = msg['name']
    if name != 'Vector fill':
        raise ValueError(f"Expected 'Vector fill', got '{name}'")

    # All fields are larger than 0
    if not all(msg[field] > 0 for field in ['requested', 'added', 'max_alloc', 'peak', 'nAlloc', 'nDealloc']):
        raise ValueError(f"Vector fill: all fields must be > 0, got {msg}")

    # added is larger than max alloc
    if msg['added'] <= msg['max_alloc']:
        raise ValueError(f"Vector fill: added ({msg['added']}) must be > max alloc ({msg['max_alloc']})")

    # nAlloc is exactly 1 larger than nDealloc
    if msg['nAlloc'] - msg['nDealloc'] != 1:
        raise ValueError(f"Vector fill: nAlloc - nDealloc must be 1, got {msg['nAlloc']} - {msg['nDealloc']} = {msg['nAlloc'] - msg['nDealloc']}")

def verify_vector_fill_again(msg, vector_fill_msg):
    """Verify conditions for 'Vector fill again' message."""
    name = msg['name']
    if name != 'Vector fill again':
        raise ValueError(f"Expected 'Vector fill again', got '{name}'")

    # All fields are larger than 0
    if not all(msg[field] > 0 for field in ['requested', 'added', 'max_alloc', 'peak', 'nAlloc', 'nDealloc']):
        raise ValueError(f"Vector fill again: all fields must be > 0, got {msg}")

    # added plus the added from "Vector fill" is larger than max alloc of "Vector fill again"
    combined_added = msg['added'] + vector_fill_msg['added']
    if combined_added <= msg['max_alloc']:
        raise ValueError(f"Vector fill again: combined added ({combined_added}) must be > max alloc ({msg['max_alloc']})")

    # nAlloc and nDealloc are equal
    if msg['nAlloc'] != msg['nDealloc']:
        raise ValueError(f"Vector fill again: nAlloc ({msg['nAlloc']}) must equal nDealloc ({msg['nDealloc']})")

def verify_nested_empty_outer_inner(msg):
    """Verify conditions for 'Nested allocation empty outer;inner unique_ptr' message."""
    name = msg['name']
    if name != 'Nested allocation empty outer;inner unique_ptr':
        raise ValueError(f"Expected 'Nested allocation empty outer;inner unique_ptr', got '{name}'")

    # requested is 4
    if msg['requested'] != 4:
        raise ValueError(f"Nested allocation empty outer;inner unique_ptr: requested must be 4, got {msg['requested']}")

    # added and peak are equal
    if msg['added'] != msg['peak']:
        raise ValueError(f"Nested allocation empty outer;inner unique_ptr: added ({msg['added']}) must equal peak ({msg['peak']})")

    # added is larger or equal to requested
    if msg['added'] < msg['requested']:
        raise ValueError(f"Nested allocation empty outer;inner unique_ptr: added ({msg['added']}) must be >= requested ({msg['requested']})")

    # nAlloc is 1 and nDealloc is 0
    if msg['nAlloc'] != 1:
        raise ValueError(f"Nested allocation empty outer;inner unique_ptr: nAlloc must be 1, got {msg['nAlloc']}")
    if msg['nDealloc'] != 0:
        raise ValueError(f"Nested allocation empty outer;inner unique_ptr: nDealloc must be 0, got {msg['nDealloc']}")

def verify_nested_empty_outer(msg, inner_msg):
    """Verify conditions for 'Nested allocation empty outer' message."""
    name = msg['name']
    if name != 'Nested allocation empty outer':
        raise ValueError(f"Expected 'Nested allocation empty outer', got '{name}'")

    # requested is 0
    if msg['requested'] != 0:
        raise ValueError(f"Nested allocation empty outer: requested must be 0, got {msg['requested']}")

    # added is negative of inner added
    if msg['added'] != -inner_msg['added']:
        raise ValueError(f"Nested allocation empty outer: added ({msg['added']}) must be -{inner_msg['added']}")

    # max alloc is 0
    if msg['max_alloc'] != 0:
        raise ValueError(f"Nested allocation empty outer: max alloc must be 0, got {msg['max_alloc']}")

    # peak is 0
    if msg['peak'] != 0:
        raise ValueError(f"Nested allocation empty outer: peak must be 0, got {msg['peak']}")

    # nAlloc is 0
    if msg['nAlloc'] != 0:
        raise ValueError(f"Nested allocation empty outer: nAlloc must be 0, got {msg['nAlloc']}")

    # nDealloc is 1
    if msg['nDealloc'] != 1:
        raise ValueError(f"Nested allocation empty outer: nDealloc must be 1, got {msg['nDealloc']}")

def verify_nested_outer_unique_ptr_inner(msg):
    """Verify conditions for 'Nested allocation outer unique_ptr;inner unique_ptr' message."""
    name = msg['name']
    if name != 'Nested allocation outer unique_ptr;inner unique_ptr':
        raise ValueError(f"Expected 'Nested allocation outer unique_ptr;inner unique_ptr', got '{name}'")

    # requested is 4
    if msg['requested'] != 4:
        raise ValueError(f"Nested allocation outer unique_ptr;inner unique_ptr: requested must be 4, got {msg['requested']}")

    # added and peak are equal
    if msg['added'] != msg['peak']:
        raise ValueError(f"Nested allocation outer unique_ptr;inner unique_ptr: added ({msg['added']}) must equal peak ({msg['peak']})")

    # added is larger or equal to requested
    if msg['added'] < msg['requested']:
        raise ValueError(f"Nested allocation outer unique_ptr;inner unique_ptr: added ({msg['added']}) must be >= requested ({msg['requested']})")

    # nAlloc is 1 and nDealloc is 0
    if msg['nAlloc'] != 1:
        raise ValueError(f"Nested allocation outer unique_ptr;inner unique_ptr: nAlloc must be 1, got {msg['nAlloc']}")
    if msg['nDealloc'] != 0:
        raise ValueError(f"Nested allocation outer unique_ptr;inner unique_ptr: nDealloc must be 0, got {msg['nDealloc']}")

def verify_nested_outer_unique_ptr(msg, inner_msg):
    """Verify conditions for 'Nested allocation outer unique_ptr' message."""
    name = msg['name']
    if name != 'Nested allocation outer unique_ptr':
        raise ValueError(f"Expected 'Nested allocation outer unique_ptr', got '{name}'")

    # requested, max alloc, and peak are larger than 0
    if msg['requested'] <= 0:
        raise ValueError(f"Nested allocation outer unique_ptr: requested must be > 0, got {msg['requested']}")
    if msg['max_alloc'] <= 0:
        raise ValueError(f"Nested allocation outer unique_ptr: max alloc must be > 0, got {msg['max_alloc']}")
    if msg['peak'] <= 0:
        raise ValueError(f"Nested allocation outer unique_ptr: peak must be > 0, got {msg['peak']}")

    # added is negative of inner added
    if msg['added'] != -inner_msg['added']:
        raise ValueError(f"Nested allocation outer unique_ptr: added ({msg['added']}) must be -{inner_msg['added']}")

    # requested and max alloc equal to inner
    if msg['requested'] != inner_msg['requested']:
        raise ValueError(f"Nested allocation outer unique_ptr: requested ({msg['requested']}) must equal inner requested ({inner_msg['requested']})")
    if msg['max_alloc'] != inner_msg['max_alloc']:
        raise ValueError(f"Nested allocation outer unique_ptr: max alloc ({msg['max_alloc']}) must equal inner max alloc ({inner_msg['max_alloc']})")

    # peak equals inner peak
    if msg['peak'] != inner_msg['peak']:
        raise ValueError(f"Nested allocation outer unique_ptr: peak ({msg['peak']}) must equal inner peak ({inner_msg['peak']})")

    # nAlloc is 1
    if msg['nAlloc'] != 1:
        raise ValueError(f"Nested allocation outer unique_ptr: nAlloc must be 1, got {msg['nAlloc']}")

    # nDealloc is 2
    if msg['nDealloc'] != 2:
        raise ValueError(f"Nested allocation outer unique_ptr: nDealloc must be 2, got {msg['nDealloc']}")

def main():
    """Main function to parse and verify IntrusiveAllocMonitor output."""
    try:
        messages = read_messages()

        # Check that we have exactly 6 messages
        if len(messages) != 6:
            raise ValueError(f"Expected exactly 6 messages, got {len(messages)}")

        # Verify each message
        verify_vector_fill(messages[0])
        verify_vector_fill_again(messages[1], messages[0])
        verify_nested_empty_outer_inner(messages[2])
        verify_nested_empty_outer(messages[3], messages[2])
        verify_nested_outer_unique_ptr_inner(messages[4])
        verify_nested_outer_unique_ptr(messages[5], messages[4])

        # All checks passed, exit silently with code 0
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
