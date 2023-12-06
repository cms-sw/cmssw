def swap(arr, a, b):
    # print('Compare (%d, %d)' % (a, b))

    if arr[a] < arr[b]:
        temp = arr[a]
        arr[a] = arr[b]
        arr[b] = temp

def merge_block(arr, offset, step, block_begin, block_end, first_n=-1):
    wire_offset = offset + block_begin
    wire_cutoff = first_n + block_begin
    wire_1 = wire_offset
    wire_2 = wire_1 + step

    # Merge block
    while wire_2 < block_end:
        # Trim
        if first_n > -1 and wire_cutoff < block_end:
            if (first_n < step or step == 1):
                wire_required = wire_offset <= wire_1 <= wire_cutoff
            else:
                wire_required = wire_offset <= wire_1 < wire_cutoff

            if not wire_required:
                break

        # Swap
        swap(arr, wire_1, wire_2)

        # Calculate new wire_1
        if step == 1:
            wire_1 = wire_2 + 1
        else:
            wire_1 = wire_1 + 1

        # Calculate new wire_2
        wire_2 = wire_1 + step


def merge_sort(arr, first_n=-1):
    arr_length = len(arr)

    # Sort
    for i in range(int(arr_length / 2)):
        swap(arr, 2 * i, 2 * i + 1)

    # Merge to mas block size
    stage = 0
    offset = 0
    step = 2
    block_size = step * 2

    # Loop block sizes
    while True:
        # print('Block size=%d begin' % (block_size))

        # Loop merge steps
        # if the offset surpases the amount of wires to keep,
        # there's no need to continue
        while True:
            # Loop blocks
            stage += 1
            block = 0
            block_begin = 0
            block_end = min(block_size, arr_length)

            while block_begin < arr_length:
                # print('Stage %d Block %d offset=%d step=%d size=%d [%d, %d] begin' % (stage, block, offset, step, block_size, block_begin, block_end))

                merge_block(arr, offset, step, block_begin, block_end, first_n=first_n)

                # Move to next block
                block += 1
                block_begin = block_end
                block_end = min(block_end + block_size, arr_length)

            # Decrease step
            if step > 2:
                # For each pass we are certain of the local min and max so skip 2 wires and reduce step
                offset = offset + 2
                step = step - 2
            elif step == 2:
                # For final pass we are certain of the global min and max so skip 1 wire and compare 1 to 1
                # the last value will be left without a partner; naturally since it's the global min
                offset = 1
                step = 1
            else:
                # Short-Circuit: Done
                break;

        # Short-Circuit: No more wires
        if block_size >= arr_length:
            break

        # Double the block size
        offset = 0
        step = block_size
        block_size = step * 2

import random
import copy

random.seed(2022)

match_count = 0
test_runs = 1

for i in range(test_runs):
    random.seed(2022 + i)
    original = [random.randint(0,120) for i in range(144)]

    print(original, len(original))

    # Osvaldo Mode
    osvaldo_input = copy.deepcopy(original)
    osvaldo_output = copy.deepcopy(osvaldo_input)
    merge_sort(osvaldo_output, first_n=4)

    # Jia Fu Mode
    jf_p1_input = copy.deepcopy(original[:32])
    jf_p2_input = copy.deepcopy(original[32:])
    jf_p1_output = copy.deepcopy(jf_p1_input)

    merge_sort(jf_p1_output, first_n=16)

    for i in range(28):
        part = jf_p2_input[i*4:(i+1)*4]
        merge_sort(part)
        #print(i, part)
        jf_p2_input[i*4:(i+1)*4] = part

    jf_input = copy.deepcopy(jf_p1_output[:16] + jf_p2_input)
    jf_output = copy.deepcopy(jf_input)
    #print(jf_output, len(jf_output))
    merge_sort(jf_output, first_n=4)

    print('Osvaldo Mode')
    print('\nInput Part 1', osvaldo_input[:32])
    print('\nInput Part 2', osvaldo_input[32:])
    print('\nInput Length',  len(osvaldo_input))
    print('\nBest', osvaldo_output[:4])
    print()
    print('Jia Fu Mode')
    print('\nInput Part 1', jf_p1_input)
    # print('\nOutput Part 1', jf_p1_output[:16])
    print('\nInput Part 2', jf_p2_input)
    print('\nInput Length', len(jf_input))
    print('\nBest', jf_output[:4])

    matches = True

    for i in range(4):
        if osvaldo_output[i] != jf_output[i] or osvaldo_output[i] != jf_output[i]:
            matches = False
            break

    if matches:
        match_count += 1

    print('\nOutputs Match', matches)

print('Match Count', match_count, 'Runs', test_runs)

