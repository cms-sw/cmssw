#!/bin/bash

# Number of runs per test (first can be treated as warm-up)
runs=6

# Threads/Streams combinations to test
thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "24:24" "32:32" "48:48" "64:64")

# Script to run
local_script_name="hlt_test.py"

# Base directory for logs
BASE_DIR="../../test_results/plain_hlt_dependence_within_socket_thread-stream-core"
mkdir -p "$BASE_DIR"

# Define NUMA CPU mapping
declare -A numa_cpu_map=(
    [6]="48 49 50 51 52 53 54 55 112 113 114 115 116 117 118 119"
    [7]="56 57 58 59 60 61 62 63 120 121 122 123 124 125 126 127"
    [5]="40 41 42 43 44 45 46 47 104 105 106 107 108 109 110 111"
    [4]="32 33 34 35 36 37 38 39 96 97 98 99 100 101 102 103"
    [3]="24 25 26 27 28 29 30 31 88 89 90 91 92 93 94 95"
    [2]="16 17 18 19 20 21 22 23 80 81 82 83 84 85 86 87"
    [1]="8 9 10 11 12 13 14 15 72 73 74 75 76 77 78 79"
    [0]="0 1 2 3 4 5 6 7 64 65 66 67 68 69 70 71"
)

# Only use nodes 6,7,5,4
ordered_numa_nodes=(6 7 5 4)

# Rebuild all_cpus with real cores first, then hyperthreads
real_cores=()
ht_cores=()
for node in "${ordered_numa_nodes[@]}"; do
    cpu_list=(${numa_cpu_map[$node]})
    real_cores+=( "${cpu_list[@]:0:8}" )
    ht_cores+=( "${cpu_list[@]:8:8}" )
done
all_cpus=( "${real_cores[@]}" "${ht_cores[@]}" )

for combo in "${thread_stream_combos[@]}"; do
    IFS=':' read -r threads streams <<< "$combo"
    total_cores=$streams  # only streams determine how many physical cores we bind to

    # Pick just enough CPUs to cover total_cores
    selected_cpus=("${all_cpus[@]:0:$total_cores}")
    cpu_list=$(IFS=,; echo "${selected_cpus[*]}")

    # Construct the test directory name
    short_cpu_tag="n${ordered_numa_nodes[0]}"

    TEST_DIR="$BASE_DIR/test_t${threads}s${streams}_${short_cpu_tag}"
    mkdir -p "$TEST_DIR"

    echo "=== Running tests with ${threads} threads, ${streams} streams, CPUs: $cpu_list ==="

    for i in $(seq 1 $runs); do
        echo "Run #$i for t${threads}s${streams} on CPU list: $cpu_list"

        export RUN_ID=$i
        export EXPERIMENT_THREADS=$threads
        export EXPERIMENT_STREAMS=$streams
        export EXPERIMENT_NAME="local_t${threads}s${streams}_${short_cpu_tag}_r${i}"
        export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"
        export THROUGHPUT_LOG_FILE="$BASE_DIR/throughputs.txt"

        # Run pinned to the CPU list
        numactl --physcpubind=$cpu_list cmsRun "$local_script_name"
    done

    echo "Completed tests for threads=$threads, streams=$streams, cpus=$cpu_list"
done

echo "All local (non-offload) tests completed!"
