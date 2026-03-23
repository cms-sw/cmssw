#!/usr/bin/env python3

import sys
import re

# Section header strings as they appear in the output
SECTION_KEYS = {
    'All allocations': 'alloc',
    'Allocated memory at the max actual moment': 'atMaxActual',
    'Added memory': 'added',
    'All deallocations': 'dealloc',
    'Memory allocation+deallocation churn': 'churn',
    'Memory allocation+deallocation churn allocations': 'churnalloc',
}


def _parse_alloc_record_line(stripped):
    """Parse 'count C requested R actual A' → dict, or None."""
    m = re.match(r'^count\s+(\d+)\s+requested\s+(\d+)\s+actual\s+(\d+)', stripped)
    if m:
        return {'count': int(m.group(1)), 'requested': int(m.group(2)),
                'actual': int(m.group(3)), 'trace': ''}
    return None


def _parse_dealloc_record_line(stripped):
    """Parse 'count C actual A' → dict, or None."""
    m = re.match(r'^count\s+(\d+)\s+actual\s+(\d+)', stripped)
    if m:
        return {'count': int(m.group(1)), 'actual': int(m.group(2)), 'trace': ''}
    return None


def totals(records, kind='alloc'):
    """Return summed count/requested/actual over a list of records."""
    if not records:
        t = {'count': 0, 'actual': 0}
        if kind != 'dealloc':
            t['requested'] = 0
        return t
    t = {
        'count': sum(r['count'] for r in records),
        'actual': sum(r['actual'] for r in records),
    }
    if kind != 'dealloc':
        t['requested'] = sum(r['requested'] for r in records)
    return t


def parse_trace_message(lines):
    """Parse one %MSG block (the raw lines between %MSG-s and %MSG).

    Returns a dict with keys: name, alloc, atMaxActual, added, dealloc, churn, churnalloc.
    Each section value is a list of record dicts.
    Returns None for 'Starting tracing' blocks.
    """
    # Handle line-wrapping by joining stripped non-empty lines for name extraction
    joined = ' '.join(l.strip() for l in lines if l.strip())

    if 'Ending tracing for' not in joined:
        return None  # 'Starting tracing' message – skip

    name_m = re.search(r'Ending tracing for "([^"]+)"', joined)
    if not name_m:
        return None
    name = name_m.group(1)

    result = {'name': name, 'alloc': [], 'atMaxActual': [], 'added': [], 'dealloc': [], 'churn': [], 'churnalloc': []}
    current_section = None
    current_record = None

    for line in lines:
        stripped = line.strip()

        # Section header?
        if stripped in SECTION_KEYS:
            if current_record is not None:
                result[current_section].append(current_record)
                current_record = None
            current_section = SECTION_KEYS[stripped]
            continue

        if current_section is None:
            continue  # lines before first section header (stack trace preamble etc.)

        # Blank line ends the current record
        if not stripped:
            if current_record is not None:
                result[current_section].append(current_record)
                current_record = None
            continue

        # Try to start a new record
        if current_section in ('alloc', 'atMaxActual', 'added', 'churn', 'churnalloc'):
            rec = _parse_alloc_record_line(stripped)
            if rec is not None:
                if current_record is not None:
                    result[current_section].append(current_record)
                current_record = rec
                continue
        elif current_section == 'dealloc':
            rec = _parse_dealloc_record_line(stripped)
            if rec is not None:
                if current_record is not None:
                    result[current_section].append(current_record)
                current_record = rec
                continue

        # Stack trace line – append to current record's trace string
        if current_record is not None:
            current_record['trace'] += stripped + '\n'

    # Flush final record
    if current_record is not None and current_section is not None:
        result[current_section].append(current_record)

    return result


def read_test_cases():
    """Read stdin and extract test cases with their IntrusiveAllocProfiler messages."""
    test_cases = []
    current = None
    collecting_msg = False
    msg_lines = []

    for raw_line in sys.stdin:
        line = raw_line.strip()

        # Start of a new test case
        if line.startswith('Test '):
            current = {'name': line[5:], 'messages': [], 'sum': None}
            collecting_msg = False
            msg_lines = []
            continue

        if current is None:
            continue

        # End of a test case
        if line.startswith('===='):
            if current is not None:
                test_cases.append(current)
            current = None
            continue

        # Start of a %MSG block
        if line.startswith('%MSG-s IntrusiveAllocProfiler:'):
            collecting_msg = True
            msg_lines = []
            continue

        # End of a %MSG block
        if line == '%MSG' and collecting_msg:
            collecting_msg = False
            msg = parse_trace_message(msg_lines)
            if msg is not None:
                current['messages'].append(msg)
            msg_lines = []
            continue

        if collecting_msg:
            msg_lines.append(raw_line)
            continue

        if line.startswith('Sum '):
            current['sum'] = int(line[4:])

    if current is not None:
        test_cases.append(current)

    return test_cases


# ---------------------------------------------------------------------------
# Per-test verifiers
# ---------------------------------------------------------------------------

def verify_vector_fill(test_case):
    """Verify the 'vector fill' test case."""
    name = test_case['name']
    if name != 'vector fill':
        raise ValueError(f"Expected test 'vector fill', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 1:
        raise ValueError(f"vector fill: expected 1 message, got {len(messages)}")

    msg = messages[0]
    if msg['name'] != 'Vector fill':
        raise ValueError(f"vector fill: expected message name 'Vector fill', got '{msg['name']}'")

    alloc_t = totals(msg['alloc'])
    added_t = totals(msg['added'])
    dealloc_t = totals(msg['dealloc'], 'dealloc')

    # Multiple allocations due to vector growth, one live buffer at end
    if alloc_t['count'] <= 1:
        raise ValueError(f"vector fill: alloc count must be > 1 (vector reallocates), got {alloc_t['count']}")

    # Exactly one allocation is still live (the final buffer)
    if added_t['count'] != 1:
        raise ValueError(f"vector fill: added count must be 1, got {added_t['count']}")

    # All intermediate buffers were freed: dealloc = alloc - added
    expected_dealloc = alloc_t['count'] - added_t['count']
    if dealloc_t['count'] != expected_dealloc:
        raise ValueError(f"vector fill: dealloc count must be {expected_dealloc}, got {dealloc_t['count']}")

    # Churn: repeated alloc+free of growing buffers
    if not msg['churn']:
        raise ValueError("vector fill: churn section must be non-empty")

    # churnalloc totals match churn totals (same alloc+dealloc pairs, different grouping)
    churn_t = totals(msg['churn'])
    churnalloc_t = totals(msg['churnalloc'])
    if churnalloc_t['count'] != churn_t['count']:
        raise ValueError(f"vector fill: churnalloc count ({churnalloc_t['count']}) must equal churn count ({churn_t['count']})")
    if churnalloc_t['actual'] != churn_t['actual']:
        raise ValueError(f"vector fill: churnalloc actual ({churnalloc_t['actual']}) must equal churn actual ({churn_t['actual']})")

    # At peak, two buffers are simultaneously live (old + new during realloc)
    atMaxActual_t = totals(msg['atMaxActual'])
    if atMaxActual_t['count'] != 2:
        raise ValueError(f"vector fill: atMaxActual count must be 2 (two buffers live at realloc peak), got {atMaxActual_t['count']}")

    if test_case['sum'] != 99980000:
        raise ValueError(f"vector fill: expected sum 99980000, got {test_case['sum']}")


def verify_vector_fill_again(test_case):
    """Verify the 'vector fill again' test case."""
    name = test_case['name']
    if name != 'vector fill again':
        raise ValueError(f"Expected test 'vector fill again', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 1:
        raise ValueError(f"vector fill again: expected 1 message, got {len(messages)}")

    msg = messages[0]
    if msg['name'] != 'Vector fill again':
        raise ValueError(f"vector fill again: expected message name 'Vector fill again', got '{msg['name']}'")

    if not msg['alloc']:
        raise ValueError("vector fill again: alloc section must be non-empty")

    added_t = totals(msg['added'])
    if added_t['count'] < 1:
        raise ValueError(f"vector fill again: added count must be >= 1, got {added_t['count']}")

    # Only one allocation in this scope (new buffer); atMaxActual count equals alloc count
    alloc_t = totals(msg['alloc'])
    atMaxActual_t = totals(msg['atMaxActual'])
    if atMaxActual_t['count'] != alloc_t['count']:
        raise ValueError(f"vector fill again: atMaxActual count ({atMaxActual_t['count']}) must equal alloc count ({alloc_t['count']})")

    # No churn: the freed buffer was allocated in a previous measurement
    if msg['churnalloc']:
        raise ValueError("vector fill again: churnalloc section must be empty (no intra-measurement alloc+free)")

    if test_case['sum'] != 199960000:
        raise ValueError(f"vector fill again: expected sum 199960000, got {test_case['sum']}")


def verify_nested_empty_outer(test_case):
    """Verify the 'nested allocation with empty outer' test case."""
    name = test_case['name']
    if name != 'nested allocation with empty outer':
        raise ValueError(f"Expected test 'nested allocation with empty outer', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 2:
        raise ValueError(f"nested allocation with empty outer: expected 2 messages, got {len(messages)}")

    inner_msg = messages[0]
    outer_msg = messages[1]

    if inner_msg['name'] != 'inner unique_ptr':
        raise ValueError(f"nested allocation with empty outer: inner message name must be 'inner unique_ptr', got '{inner_msg['name']}'")
    if outer_msg['name'] != 'Nested allocation empty outer':
        raise ValueError(f"nested allocation with empty outer: outer message name must be 'Nested allocation empty outer', got '{outer_msg['name']}'")

    # Inner: one allocation (new int(42)), still alive at inner measurement end
    inner_alloc_t = totals(inner_msg['alloc'])
    inner_added_t = totals(inner_msg['added'])

    if inner_alloc_t['count'] != 1:
        raise ValueError(f"nested allocation with empty outer (inner): alloc count must be 1, got {inner_alloc_t['count']}")
    if inner_alloc_t['requested'] != 4:
        raise ValueError(f"nested allocation with empty outer (inner): alloc requested must be 4, got {inner_alloc_t['requested']}")
    if inner_added_t['count'] != 1:
        raise ValueError(f"nested allocation with empty outer (inner): added count must be 1, got {inner_added_t['count']}")
    if inner_added_t['requested'] != 4:
        raise ValueError(f"nested allocation with empty outer (inner): added requested must be 4, got {inner_added_t['requested']}")
    if inner_msg['dealloc']:
        raise ValueError(f"nested allocation with empty outer (inner): dealloc section must be empty")
    if inner_msg['churn']:
        raise ValueError(f"nested allocation with empty outer (inner): churn section must be empty")

    # Single int allocation, live the whole inner scope: atMaxActual equals alloc
    inner_atMaxActual_t = totals(inner_msg['atMaxActual'])
    if inner_atMaxActual_t['count'] != inner_alloc_t['count']:
        raise ValueError(f"nested allocation with empty outer (inner): atMaxActual count ({inner_atMaxActual_t['count']}) must equal alloc count ({inner_alloc_t['count']})")
    if inner_msg['churnalloc']:
        raise ValueError("nested allocation with empty outer (inner): churnalloc section must be empty")

    # Outer: no allocations of its own; the one deallocation is the inner allocation freed here
    if outer_msg['alloc']:
        raise ValueError("nested allocation with empty outer (outer): alloc section must be empty")
    if outer_msg['added']:
        raise ValueError("nested allocation with empty outer (outer): added section must be empty")

    outer_dealloc_t = totals(outer_msg['dealloc'], 'dealloc')
    if outer_dealloc_t['count'] != 1:
        raise ValueError(f"nested allocation with empty outer (outer): dealloc count must be 1, got {outer_dealloc_t['count']}")

    # The same allocation size flows from inner alloc to outer dealloc
    if inner_alloc_t['actual'] != outer_dealloc_t['actual']:
        raise ValueError(
            f"nested allocation with empty outer: inner alloc actual ({inner_alloc_t['actual']}) "
            f"must equal outer dealloc actual ({outer_dealloc_t['actual']})"
        )

    if outer_msg['churn']:
        raise ValueError("nested allocation with empty outer (outer): churn section must be empty")
    if outer_msg['atMaxActual']:
        raise ValueError("nested allocation with empty outer (outer): atMaxActual section must be empty")
    if outer_msg['churnalloc']:
        raise ValueError("nested allocation with empty outer (outer): churnalloc section must be empty")

    if test_case['sum'] != 42:
        raise ValueError(f"nested allocation with empty outer: expected sum 42, got {test_case['sum']}")


def verify_nested_allocation(test_case):
    """Verify the 'nested allocation' test case."""
    name = test_case['name']
    if name != 'nested allocation':
        raise ValueError(f"Expected test 'nested allocation', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 2:
        raise ValueError(f"nested allocation: expected 2 messages, got {len(messages)}")

    inner_msg = messages[0]
    outer_msg = messages[1]

    if inner_msg['name'] != 'inner unique_ptr':
        raise ValueError(f"nested allocation (inner): expected name 'inner unique_ptr', got '{inner_msg['name']}'")
    if outer_msg['name'] != 'Nested allocation outer unique_ptr':
        raise ValueError(f"nested allocation (outer): expected name 'Nested allocation outer unique_ptr', got '{outer_msg['name']}'")

    # Inner: one allocation (new int(42)), alive at inner measurement end
    inner_alloc_t = totals(inner_msg['alloc'])
    inner_added_t = totals(inner_msg['added'])

    if inner_alloc_t['count'] != 1:
        raise ValueError(f"nested allocation (inner): alloc count must be 1, got {inner_alloc_t['count']}")
    if inner_alloc_t['requested'] != 4:
        raise ValueError(f"nested allocation (inner): alloc requested must be 4, got {inner_alloc_t['requested']}")
    if inner_added_t['count'] != 1:
        raise ValueError(f"nested allocation (inner): added count must be 1, got {inner_added_t['count']}")
    if inner_msg['dealloc']:
        raise ValueError("nested allocation (inner): dealloc section must be empty")
    if inner_msg['churn']:
        raise ValueError("nested allocation (inner): churn section must be empty")

    # Single int allocation, live for the whole inner scope: atMaxActual equals alloc
    inner_atMaxActual_t = totals(inner_msg['atMaxActual'])
    if inner_atMaxActual_t['count'] != inner_alloc_t['count']:
        raise ValueError(f"nested allocation (inner): atMaxActual count ({inner_atMaxActual_t['count']}) must equal alloc count ({inner_alloc_t['count']})")
    if inner_msg['churnalloc']:
        raise ValueError("nested allocation (inner): churnalloc section must be empty")

    # Outer: allocates ptr1, added is empty (both ptr1 and ptr2 freed by end of measurement),
    # dealloc has ptr1 (freed in outer) + ptr2 (attributed from inner), churn has ptr1 (alloc+free in outer)
    outer_alloc_t = totals(outer_msg['alloc'])
    outer_added_t = totals(outer_msg['added'])
    outer_dealloc_t = totals(outer_msg['dealloc'], 'dealloc')
    outer_churn_t = totals(outer_msg['churn'])

    if outer_alloc_t['count'] < 1:
        raise ValueError(f"nested allocation (outer): alloc count must be >= 1, got {outer_alloc_t['count']}")
    if outer_alloc_t['requested'] != 4:
        raise ValueError(f"nested allocation (outer): alloc requested must be 4 (ptr1), got {outer_alloc_t['requested']}")

    if outer_added_t['count'] != 0:
        raise ValueError(f"nested allocation (outer): added must be empty (all freed), got count {outer_added_t['count']}")

    # dealloc count = outer alloc + inner alloc (ptr2 attributed up)
    expected_dealloc = outer_alloc_t['count'] + inner_alloc_t['count']
    if outer_dealloc_t['count'] != expected_dealloc:
        raise ValueError(f"nested allocation (outer): dealloc count must be {expected_dealloc}, got {outer_dealloc_t['count']}")

    # churn count = outer alloc count (ptr1 was allocated and freed within this measurement)
    if outer_churn_t['count'] != outer_alloc_t['count']:
        raise ValueError(f"nested allocation (outer): churn count must equal outer alloc count ({outer_alloc_t['count']}), got {outer_churn_t['count']}")

    # ptr1 (only outer alloc) was live for the full outer scope: atMaxActual count equals alloc count
    outer_atMaxActual_t = totals(outer_msg['atMaxActual'])
    if outer_atMaxActual_t['count'] != outer_alloc_t['count']:
        raise ValueError(f"nested allocation (outer): atMaxActual count ({outer_atMaxActual_t['count']}) must equal alloc count ({outer_alloc_t['count']})")

    # churnalloc totals match churn totals
    outer_churnalloc_t = totals(outer_msg['churnalloc'])
    if outer_churnalloc_t['count'] != outer_churn_t['count']:
        raise ValueError(f"nested allocation (outer): churnalloc count ({outer_churnalloc_t['count']}) must equal churn count ({outer_churn_t['count']})")
    if outer_churnalloc_t['actual'] != outer_churn_t['actual']:
        raise ValueError(f"nested allocation (outer): churnalloc actual ({outer_churnalloc_t['actual']}) must equal churn actual ({outer_churn_t['actual']})")

    if test_case['sum'] != 84:
        raise ValueError(f"nested allocation: expected sum 84, got {test_case['sum']}")


def verify_nested_with_string_messages(test_case):
    """Verify the 'nested with string messages' test case."""
    name = test_case['name']
    if name != 'nested with string messages':
        raise ValueError(f"Expected test 'nested with string messages', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 5:
        raise ValueError(f"nested with string messages: expected 5 messages, got {len(messages)}")

    def verify_simple_inner(msg, expected_name, msg_id):
        """Verify an inner measurement that allocates a single int and keeps it alive."""
        if msg['name'] != expected_name:
            raise ValueError(f"nested with string messages (msg {msg_id}): expected name '{expected_name}', got '{msg['name']}'")

        alloc_t = totals(msg['alloc'])
        added_t = totals(msg['added'])

        if alloc_t['count'] != 1:
            raise ValueError(f"nested with string messages (msg {msg_id}): alloc count must be 1, got {alloc_t['count']}")
        if alloc_t['requested'] != 4:
            raise ValueError(f"nested with string messages (msg {msg_id}): alloc requested must be 4, got {alloc_t['requested']}")
        if added_t['count'] != 1:
            raise ValueError(f"nested with string messages (msg {msg_id}): added count must be 1, got {added_t['count']}")
        if added_t['requested'] != 4:
            raise ValueError(f"nested with string messages (msg {msg_id}): added requested must be 4, got {added_t['requested']}")
        if msg['dealloc']:
            raise ValueError(f"nested with string messages (msg {msg_id}): dealloc section must be empty")
        if msg['churn']:
            raise ValueError(f"nested with string messages (msg {msg_id}): churn section must be empty")

        # Single int allocation: atMaxActual equals alloc, no churnalloc
        atMaxActual_t = totals(msg['atMaxActual'])
        if atMaxActual_t['count'] != 1:
            raise ValueError(f"nested with string messages (msg {msg_id}): atMaxActual count must be 1, got {atMaxActual_t['count']}")
        if atMaxActual_t['requested'] != 4:
            raise ValueError(f"nested with string messages (msg {msg_id}): atMaxActual requested must be 4, got {atMaxActual_t['requested']}")
        if msg['churnalloc']:
            raise ValueError(f"nested with string messages (msg {msg_id}): churnalloc section must be empty")

    # Messages 0-2: simple inner measurements (ptr2, ptr3, ptr4)
    verify_simple_inner(messages[0], 'inner unique_ptr', 0)
    verify_simple_inner(messages[1], 'another inner unique_ptr', 1)
    verify_simple_inner(messages[2], 'inner unique_ptr', 2)

    # Message 3: 'inner empty' – allocates a string for the sub-measurement name,
    # that string is then freed in the same scope. ptr4 is freed in the outer scope.
    inner_empty = messages[3]
    if inner_empty['name'] != 'inner empty':
        raise ValueError(f"nested with string messages (msg 3): expected name 'inner empty', got '{inner_empty['name']}'")

    if not inner_empty['alloc']:
        raise ValueError("nested with string messages (msg 3): alloc section must be non-empty (string allocation)")

    inner_empty_alloc_t = totals(inner_empty['alloc'])
    inner_empty_dealloc_t = totals(inner_empty['dealloc'], 'dealloc')

    if inner_empty['added']:
        raise ValueError("nested with string messages (msg 3): added section must be empty")

    if not inner_empty['dealloc']:
        raise ValueError("nested with string messages (msg 3): dealloc section must be non-empty")

    # All allocations observed in 'inner empty' were also deallocated within it
    if inner_empty_alloc_t['count'] != inner_empty_dealloc_t['count']:
        raise ValueError(
            f"nested with string messages (msg 3): alloc count ({inner_empty_alloc_t['count']}) "
            f"must equal dealloc count ({inner_empty_dealloc_t['count']})"
        )

    if not inner_empty['churn']:
        raise ValueError("nested with string messages (msg 3): churn section must be non-empty")

    # churnalloc totals match churn totals
    inner_empty_churn_t = totals(inner_empty['churn'])
    inner_empty_churnalloc_t = totals(inner_empty['churnalloc'])
    if inner_empty_churnalloc_t['count'] != inner_empty_churn_t['count']:
        raise ValueError(
            f"nested with string messages (msg 3): churnalloc count ({inner_empty_churnalloc_t['count']}) "
            f"must equal churn count ({inner_empty_churn_t['count']})"
        )
    if inner_empty_churnalloc_t['actual'] != inner_empty_churn_t['actual']:
        raise ValueError(
            f"nested with string messages (msg 3): churnalloc actual ({inner_empty_churnalloc_t['actual']}) "
            f"must equal churn actual ({inner_empty_churn_t['actual']})"
        )

    # One allocation (string name): atMaxActual count equals alloc count
    inner_empty_atMaxActual_t = totals(inner_empty['atMaxActual'])
    if inner_empty_atMaxActual_t['count'] != inner_empty_alloc_t['count']:
        raise ValueError(
            f"nested with string messages (msg 3): atMaxActual count ({inner_empty_atMaxActual_t['count']}) "
            f"must equal alloc count ({inner_empty_alloc_t['count']})"
        )

    # Message 4: outer measurement – ptr1 was allocated here; all 4 ptrs are freed here
    outer_msg = messages[4]
    if outer_msg['name'] != 'Nested allocation with string messages outer unique_ptr':
        raise ValueError(
            f"nested with string messages (msg 4): expected name "
            f"'Nested allocation with string messages outer unique_ptr', got '{outer_msg['name']}'"
        )

    if not outer_msg['alloc']:
        raise ValueError("nested with string messages (msg 4): alloc section must be non-empty")

    if outer_msg['added']:
        raise ValueError("nested with string messages (msg 4): added section must be empty (all ptrs freed)")

    outer_dealloc_t = totals(outer_msg['dealloc'], 'dealloc')
    if outer_dealloc_t['count'] < 4:
        raise ValueError(
            f"nested with string messages (msg 4): dealloc count must be >= 4 (four ptrs freed), "
            f"got {outer_dealloc_t['count']}"
        )

    if not outer_msg['churn']:
        raise ValueError("nested with string messages (msg 4): churn section must be non-empty")

    # churnalloc totals match churn totals
    outer_alloc_t = totals(outer_msg['alloc'])
    outer_churn_t = totals(outer_msg['churn'])
    outer_churnalloc_t = totals(outer_msg['churnalloc'])
    if outer_churnalloc_t['count'] != outer_churn_t['count']:
        raise ValueError(
            f"nested with string messages (msg 4): churnalloc count ({outer_churnalloc_t['count']}) "
            f"must equal churn count ({outer_churn_t['count']})"
        )
    if outer_churnalloc_t['actual'] != outer_churn_t['actual']:
        raise ValueError(
            f"nested with string messages (msg 4): churnalloc actual ({outer_churnalloc_t['actual']}) "
            f"must equal churn actual ({outer_churn_t['actual']})"
        )

    # atMaxActual is non-empty and within alloc count
    if not outer_msg['atMaxActual']:
        raise ValueError("nested with string messages (msg 4): atMaxActual section must be non-empty")
    outer_atMaxActual_t = totals(outer_msg['atMaxActual'])
    if outer_atMaxActual_t['count'] > outer_alloc_t['count']:
        raise ValueError(
            f"nested with string messages (msg 4): atMaxActual count ({outer_atMaxActual_t['count']}) "
            f"must be <= alloc count ({outer_alloc_t['count']})"
        )

    if test_case['sum'] != 433:
        raise ValueError(f"nested with string messages: expected sum 433, got {test_case['sum']}")


def verify_nested_churning(test_case):
    """Verify the 'nested churning' test case."""
    name = test_case['name']
    if name != 'nested churning':
        raise ValueError(f"Expected test 'nested churning', got '{name}'")

    messages = test_case['messages']
    if len(messages) != 1:
        raise ValueError(f"nested churning: expected 1 message, got {len(messages)}")

    msg = messages[0]
    if msg['name'] != 'Nested churning':
        raise ValueError(f"nested churning: expected message name 'Nested churning', got '{msg['name']}'")

    alloc_t = totals(msg['alloc'])
    dealloc_t = totals(msg['dealloc'], 'dealloc')

    # vec is destroyed before the guard (reverse declaration order), so all
    # memory is freed within the scope: added must be empty
    if msg['added']:
        raise ValueError("nested churning: added section must be empty (vec freed before guard)")

    # Multiple allocations due to vector growth
    if alloc_t['count'] <= 1:
        raise ValueError(f"nested churning: alloc count must be > 1 (vector reallocates), got {alloc_t['count']}")

    # All allocations were freed: dealloc count equals alloc count
    if dealloc_t['count'] != alloc_t['count']:
        raise ValueError(
            f"nested churning: dealloc count ({dealloc_t['count']}) must equal "
            f"alloc count ({alloc_t['count']})"
        )

    # Churn section must be non-empty (intermediate buffers allocated and freed)
    if not msg['churn']:
        raise ValueError("nested churning: churn section must be non-empty")

    # churnalloc totals match churn totals
    churn_t = totals(msg['churn'])
    churnalloc_t = totals(msg['churnalloc'])
    if churnalloc_t['count'] != churn_t['count']:
        raise ValueError(f"nested churning: churnalloc count ({churnalloc_t['count']}) must equal churn count ({churn_t['count']})")
    if churnalloc_t['actual'] != churn_t['actual']:
        raise ValueError(f"nested churning: churnalloc actual ({churnalloc_t['actual']}) must equal churn actual ({churn_t['actual']})")

    # nestedChurn() must appear in at least one churn stack trace
    if not any('nestedChurn' in r['trace'] for r in msg['churn']):
        raise ValueError("nested churning: 'nestedChurn' must appear in at least one churn stack trace")

    # nestedChurn() must appear in at least one churnalloc stack trace
    if not any('nestedChurn' in r['trace'] for r in msg['churnalloc']):
        raise ValueError("nested churning: 'nestedChurn' must appear in at least one churnalloc stack trace")

    # At peak, two buffers are simultaneously live (old + new during realloc)
    atMaxActual_t = totals(msg['atMaxActual'])
    if atMaxActual_t['count'] != 2:
        raise ValueError(f"nested churning: atMaxActual count must be 2 (two buffers live at realloc peak), got {atMaxActual_t['count']}")

    if test_case['sum'] != 149935000:
        raise ValueError(f"nested churning: expected sum 149935000, got {test_case['sum']}")


def main():
    """Main: parse and verify IntrusiveAllocProfiler output from stdin."""
    try:
        test_cases = read_test_cases()

        if len(test_cases) != 6:
            raise ValueError(f"Expected exactly 6 test cases, got {len(test_cases)}")

        verify_vector_fill(test_cases[0])
        verify_vector_fill_again(test_cases[1])
        verify_nested_empty_outer(test_cases[2])
        verify_nested_allocation(test_cases[3])
        verify_nested_with_string_messages(test_cases[4])
        verify_nested_churning(test_cases[5])

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
