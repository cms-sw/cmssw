#!/usr/bin/env python3

import sys
import re

def parse_message(lines):
    """Parse a IntrusiveAllocMonitor message block and extract field values."""
    # First line: "measured: requested X added Y max alloc Z peak W nAlloc N nDealloc M"
    # Note: added can be negative
    pattern = r'^measured:\s+requested\s+(?P<requested>\d+)\s+added\s+(?P<added>-?\d+)\s+max alloc\s+(?P<max_alloc>\d+)\s+peak\s+(?P<peak>\d+)\s+nAlloc\s+(?P<nAlloc>\d+)\s+nDealloc\s+(?P<nDealloc>\d+)$'
    match = re.match(pattern, lines[0].strip())
    if not match:
        raise ValueError(f"Malformed message line: {lines[0]}")

    data = match.groupdict()
    # Convert numeric fields to integers
    for key in ['requested', 'added', 'max_alloc', 'peak', 'nAlloc', 'nDealloc']:
        data[key] = int(data[key])

    # Parse nested labels from lines like "[0] inner unique_ptr"
    data['labels'] = []
    data['stringAdded'] = None
    data['stringAlloc'] = None
    for line in lines[1:]:
        label_match = re.match(r'^\[(\d+)\]\s+(.+)$', line.strip())
        if label_match:
            data['labels'].append(label_match.group(2))
        else:
            # Parse optional line: "This includes at least X bytes in Y allocations from string names in nested measurements"
            string_match = re.match(r'^This includes at least (\d+) bytes in (\d+) allocations? from string names in nested measurements$', line.strip())
            if string_match:
                data['stringAdded'] = int(string_match.group(1))
                data['stringAlloc'] = int(string_match.group(2))

    return data

def read_test_cases():
    """Read stdin and extract test cases with their IntrusiveAllocMonitor messages."""
    test_cases = []

    for line in sys.stdin:
        line = line.strip()

        # Look for test case start
        if line.startswith('Test '):
            test_name = line[5:]  # Remove "Test " prefix
            messages = []
            sum_value = None

            # Read until we hit the separator
            for line in sys.stdin:
                line_stripped = line.strip()
                if line_stripped.startswith('===='):
                    break

                if line_stripped == '%MSG-s IntrusiveAllocMonitor:':
                    # Collect message lines
                    msg_lines = []
                    for line in sys.stdin:
                        if line.strip() == '%MSG':
                            break
                        msg_lines.append(line)

                    if msg_lines:
                        msg = parse_message(msg_lines)
                        messages.append(msg)
                elif line_stripped.startswith('Sum '):
                    sum_value = int(line_stripped[4:])

            test_cases.append({'name': test_name, 'messages': messages, 'sum': sum_value})

    return test_cases

def verify_vector_fill(test_case):
    """Verify conditions for 'vector fill' test case."""
    name = test_case['name']
    if name != 'vector fill':
        raise ValueError(f"Expected 'vector fill', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 1:
        raise ValueError(f"vector fill: expected 1 message, got {len(messages)}")

    msg = messages[0]

    # Check labels
    expected_labels = ['Vector fill']
    if msg['labels'] != expected_labels:
        raise ValueError(f"vector fill: expected labels {expected_labels}, got {msg['labels']}")

    # All fields are larger than 0
    if not all(msg[field] > 0 for field in ['requested', 'added', 'max_alloc', 'peak', 'nAlloc', 'nDealloc']):
        raise ValueError(f"vector fill: all fields must be > 0, got {msg}")

    # added is larger than max alloc
    if msg['added'] <= msg['max_alloc']:
        raise ValueError(f"vector fill: added ({msg['added']}) must be > max alloc ({msg['max_alloc']})")

    # nAlloc is exactly 1 larger than nDealloc
    if msg['nAlloc'] - msg['nDealloc'] != 1:
        raise ValueError(f"vector fill: nAlloc - nDealloc must be 1, got {msg['nAlloc']} - {msg['nDealloc']} = {msg['nAlloc'] - msg['nDealloc']}")

    # Verify sum
    if test_case['sum'] != 99980000:
        raise ValueError(f"vector fill: expected sum 99980000, got {test_case['sum']}")

def verify_vector_fill_again(test_case, vector_fill_test):
    """Verify conditions for 'vector fill again' test case."""
    name = test_case['name']
    if name != 'vector fill again':
        raise ValueError(f"Expected 'vector fill again', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 1:
        raise ValueError(f"vector fill again: expected 1 message, got {len(messages)}")

    msg = messages[0]
    vector_fill_msg = vector_fill_test['messages'][0]

    # Check labels
    expected_labels = ['Vector fill again']
    if msg['labels'] != expected_labels:
        raise ValueError(f"vector fill again: expected labels {expected_labels}, got {msg['labels']}")

    # All fields are larger than 0
    if not all(msg[field] > 0 for field in ['requested', 'added', 'max_alloc', 'peak', 'nAlloc', 'nDealloc']):
        raise ValueError(f"vector fill again: all fields must be > 0, got {msg}")

    # added plus the added from "vector fill" is larger than max alloc of "vector fill again"
    combined_added = msg['added'] + vector_fill_msg['added']
    if combined_added <= msg['max_alloc']:
        raise ValueError(f"vector fill again: combined added ({combined_added}) must be > max alloc ({msg['max_alloc']})")

    # nAlloc and nDealloc are equal
    if msg['nAlloc'] != msg['nDealloc']:
        raise ValueError(f"vector fill again: nAlloc ({msg['nAlloc']}) must equal nDealloc ({msg['nDealloc']})")

    # Verify sum
    if test_case['sum'] != 199960000:
        raise ValueError(f"vector fill again: expected sum 199960000, got {test_case['sum']}")

def verify_nested_empty_outer(test_case):
    """Verify conditions for 'nested allocation with empty outer' test case."""
    name = test_case['name']
    if name != 'nested allocation with empty outer':
        raise ValueError(f"Expected 'nested allocation with empty outer', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 2:
        raise ValueError(f"nested allocation with empty outer: expected 2 messages, got {len(messages)}")

    inner_msg = messages[0]
    outer_msg = messages[1]

    # Verify inner message
    expected_inner_labels = ['inner unique_ptr', 'Nested allocation empty outer']
    if inner_msg['labels'] != expected_inner_labels:
        raise ValueError(f"nested allocation with empty outer (inner): expected labels {expected_inner_labels}, got {inner_msg['labels']}")

    # requested is 4
    if inner_msg['requested'] != 4:
        raise ValueError(f"nested allocation with empty outer (inner): requested must be 4, got {inner_msg['requested']}")

    # added and peak are equal
    if inner_msg['added'] != inner_msg['peak']:
        raise ValueError(f"nested allocation with empty outer (inner): added ({inner_msg['added']}) must equal peak ({inner_msg['peak']})")

    # added is larger or equal to requested
    if inner_msg['added'] < inner_msg['requested']:
        raise ValueError(f"nested allocation with empty outer (inner): added ({inner_msg['added']}) must be >= requested ({inner_msg['requested']})")

    # nAlloc is 1 and nDealloc is 0
    if inner_msg['nAlloc'] != 1:
        raise ValueError(f"nested allocation with empty outer (inner): nAlloc must be 1, got {inner_msg['nAlloc']}")
    if inner_msg['nDealloc'] != 0:
        raise ValueError(f"nested allocation with empty outer (inner): nDealloc must be 0, got {inner_msg['nDealloc']}")

    # Verify outer message
    expected_outer_labels = ['Nested allocation empty outer']
    if outer_msg['labels'] != expected_outer_labels:
        raise ValueError(f"nested allocation with empty outer (outer): expected labels {expected_outer_labels}, got {outer_msg['labels']}")

    # requested is 0
    if outer_msg['requested'] != 0:
        raise ValueError(f"nested allocation with empty outer (outer): requested must be 0, got {outer_msg['requested']}")

    # added is negative of inner added
    if outer_msg['added'] != -inner_msg['added']:
        raise ValueError(f"nested allocation with empty outer (outer): added ({outer_msg['added']}) must be -{inner_msg['added']}")

    # max alloc is 0
    if outer_msg['max_alloc'] != 0:
        raise ValueError(f"nested allocation with empty outer (outer): max alloc must be 0, got {outer_msg['max_alloc']}")

    # peak is 0
    if outer_msg['peak'] != 0:
        raise ValueError(f"nested allocation with empty outer (outer): peak must be 0, got {outer_msg['peak']}")

    # nAlloc is 0
    if outer_msg['nAlloc'] != 0:
        raise ValueError(f"nested allocation with empty outer (outer): nAlloc must be 0, got {outer_msg['nAlloc']}")

    # nDealloc is 1
    if outer_msg['nDealloc'] != 1:
        raise ValueError(f"nested allocation with empty outer (outer): nDealloc must be 1, got {outer_msg['nDealloc']}")

    # Verify sum
    if test_case['sum'] != 42:
        raise ValueError(f"nested allocation with empty outer: expected sum 42, got {test_case['sum']}")

def verify_nested_allocation(test_case):
    """Verify conditions for 'nested allocation' test case."""
    name = test_case['name']
    if name != 'nested allocation':
        raise ValueError(f"Expected 'nested allocation', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 2:
        raise ValueError(f"nested allocation: expected 2 messages, got {len(messages)}")

    inner_msg = messages[0]
    outer_msg = messages[1]

    # Verify inner message
    expected_inner_labels = ['inner unique_ptr', 'Nested allocation outer unique_ptr']
    if inner_msg['labels'] != expected_inner_labels:
        raise ValueError(f"nested allocation (inner): expected labels {expected_inner_labels}, got {inner_msg['labels']}")

    # requested is 4
    if inner_msg['requested'] != 4:
        raise ValueError(f"nested allocation (inner): requested must be 4, got {inner_msg['requested']}")

    # added and peak are equal
    if inner_msg['added'] != inner_msg['peak']:
        raise ValueError(f"nested allocation (inner): added ({inner_msg['added']}) must equal peak ({inner_msg['peak']})")

    # added is larger or equal to requested
    if inner_msg['added'] < inner_msg['requested']:
        raise ValueError(f"nested allocation (inner): added ({inner_msg['added']}) must be >= requested ({inner_msg['requested']})")

    # nAlloc is 1 and nDealloc is 0
    if inner_msg['nAlloc'] != 1:
        raise ValueError(f"nested allocation (inner): nAlloc must be 1, got {inner_msg['nAlloc']}")
    if inner_msg['nDealloc'] != 0:
        raise ValueError(f"nested allocation (inner): nDealloc must be 0, got {inner_msg['nDealloc']}")

    # Verify outer message
    expected_outer_labels = ['Nested allocation outer unique_ptr']
    if outer_msg['labels'] != expected_outer_labels:
        raise ValueError(f"nested allocation (outer): expected labels {expected_outer_labels}, got {outer_msg['labels']}")

    # requested, max alloc, and peak are larger than 0
    if outer_msg['requested'] <= 0:
        raise ValueError(f"nested allocation (outer): requested must be > 0, got {outer_msg['requested']}")
    if outer_msg['max_alloc'] <= 0:
        raise ValueError(f"nested allocation (outer): max alloc must be > 0, got {outer_msg['max_alloc']}")
    if outer_msg['peak'] <= 0:
        raise ValueError(f"nested allocation (outer): peak must be > 0, got {outer_msg['peak']}")

    # added is negative of inner added
    if outer_msg['added'] != -inner_msg['added']:
        raise ValueError(f"nested allocation (outer): added ({outer_msg['added']}) must be -{inner_msg['added']}")

    # requested and max alloc equal to inner
    if outer_msg['requested'] != inner_msg['requested']:
        raise ValueError(f"nested allocation (outer): requested ({outer_msg['requested']}) must equal inner requested ({inner_msg['requested']})")
    if outer_msg['max_alloc'] != inner_msg['max_alloc']:
        raise ValueError(f"nested allocation (outer): max alloc ({outer_msg['max_alloc']}) must equal inner max alloc ({inner_msg['max_alloc']})")

    # peak equals inner peak
    if outer_msg['peak'] != inner_msg['peak']:
        raise ValueError(f"nested allocation (outer): peak ({outer_msg['peak']}) must equal inner peak ({inner_msg['peak']})")

    # nAlloc is 1
    if outer_msg['nAlloc'] != 1:
        raise ValueError(f"nested allocation (outer): nAlloc must be 1, got {outer_msg['nAlloc']}")

    # nDealloc is 2
    if outer_msg['nDealloc'] != 2:
        raise ValueError(f"nested allocation (outer): nDealloc must be 2, got {outer_msg['nDealloc']}")

    # Verify sum
    if test_case['sum'] != 84:
        raise ValueError(f"nested allocation: expected sum 84, got {test_case['sum']}")

def verify_nested_with_string_messages(test_case):
    """Verify conditions for 'nested with string messages' test case."""
    def verify_inner_allocation(msg, expected_labels, msg_id):
        """Verify conditions for an inner allocation message.
        - requested is 4
        - added is larger or equal than requested
        - peak and added are equal
        - 1 allocation, 0 deallocations
        """
        if msg['labels'] != expected_labels:
            raise ValueError(f"nested with string messages (msg {msg_id}): expected labels {expected_labels}, got {msg['labels']}")

        if msg['requested'] != 4:
            raise ValueError(f"nested with string messages (msg {msg_id}): requested must be 4, got {msg['requested']}")

        if msg['added'] < msg['requested']:
            raise ValueError(f"nested with string messages (msg {msg_id}): added ({msg['added']}) must be >= requested ({msg['requested']})")

        if msg['added'] != msg['peak']:
            raise ValueError(f"nested with string messages (msg {msg_id}): added ({msg['added']}) must equal peak ({msg['peak']})")

        if msg['nAlloc'] != 1:
            raise ValueError(f"nested with string messages (msg {msg_id}): nAlloc must be 1, got {msg['nAlloc']}")
        if msg['nDealloc'] != 0:
            raise ValueError(f"nested with string messages (msg {msg_id}): nDealloc must be 0, got {msg['nDealloc']}")

    name = test_case['name']
    if name != 'nested with string messages':
        raise ValueError(f"Expected 'nested with string messages', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 5:
        raise ValueError(f"nested with string messages: expected 5 messages, got {len(messages)}")

    # Verify first inner message
    verify_inner_allocation(
        messages[0],
        ['inner unique_ptr', 'Nested allocation with string messages outer unique_ptr'],
        0
    )

    # Verify second inner message
    verify_inner_allocation(
        messages[1],
        ['another inner unique_ptr', 'Nested allocation with string messages outer unique_ptr'],
        1
    )

    # Verify third inner message
    verify_inner_allocation(
        messages[2],
        ['inner unique_ptr', 'inner empty', 'Nested allocation with string messages outer unique_ptr'],
        2
    )

    # Verify fourth message (inner empty)
    msg3 = messages[3]
    expected_labels_3 = ['inner empty', 'Nested allocation with string messages outer unique_ptr']
    if msg3['labels'] != expected_labels_3:
        raise ValueError(f"nested with string messages (msg 3): expected labels {expected_labels_3}, got {msg3['labels']}")

    if msg3['added'] != 0:
        raise ValueError(f"nested with string messages (msg 3): added must be 0, got {msg3['added']}")

    if msg3['nAlloc'] != msg3['nDealloc']:
        raise ValueError(f"nested with string messages (msg 3): nAlloc ({msg3['nAlloc']}) must equal nDealloc ({msg3['nDealloc']})")

    if msg3['stringAlloc'] is not None:
        if msg3['nAlloc'] < msg3['stringAlloc']:
            raise ValueError(f"nested with string messages (msg 3): nAlloc ({msg3['nAlloc']}) must be >= stringAlloc ({msg3['stringAlloc']})")

    if msg3['stringAdded'] is not None:
        if msg3['requested'] < msg3['stringAdded']:
            raise ValueError(f"nested with string messages (msg 3): requested ({msg3['requested']}) must be >= stringAdded ({msg3['stringAdded']})")


    # Verify fifth message (outer)
    msg4 = messages[4]
    expected_labels_4 = ['Nested allocation with string messages outer unique_ptr']
    if msg4['labels'] != expected_labels_4:
        raise ValueError(f"nested with string messages (msg 4): expected labels {expected_labels_4}, got {msg4['labels']}")

    if msg4['stringAlloc'] is not None:
        if msg4['stringAlloc'] < 3:
            raise ValueError(f"nested with string messages (msg 4): stringAlloc must be >= 3, got {msg4['stringAlloc']}")

    if msg4['nDealloc'] - msg4['nAlloc'] != 3:
        raise ValueError(f"nested with string messages (msg 4): nDealloc - nAlloc must be 3, got {msg4['nDealloc']} - {msg4['nAlloc']} = {msg4['nDealloc'] - msg4['nAlloc']}")

    # Verify sum
    if test_case['sum'] != 433:
        raise ValueError(f"nested with string messages: expected sum 433, got {test_case['sum']}")

def main():
    """Main function to parse and verify IntrusiveAllocMonitor output."""
    try:
        test_cases = read_test_cases()

        # Check that we have exactly 5 test cases
        if len(test_cases) != 5:
            raise ValueError(f"Expected exactly 5 test cases, got {len(test_cases)}")

        # Verify each test case
        verify_vector_fill(test_cases[0])
        verify_vector_fill_again(test_cases[1], test_cases[0])
        verify_nested_empty_outer(test_cases[2])
        verify_nested_allocation(test_cases[3])
        verify_nested_with_string_messages(test_cases[4])

        # All checks passed, exit silently with code 0
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
