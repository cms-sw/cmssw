#!/bin/bash

# Number of runs per test (first can be treated as warm-up)
runs=5

# Threads/Streams combinations to test
thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "32:32" "64:64" "96:96" "128:128")

# NUMA node configurations (can be single or multiple nodes like "6", "6-7", "45-67")
# numa_nodes=("6" "6,7")

# Script to run
local_script_name="hlt_test.py"

# Base directory for logs
BASE_DIR="../../test_results/local_thread_numa"
mkdir -p "$BASE_DIR"

for combo in "${thread_stream_combos[@]}"; do
    IFS=':' read -r threads streams <<< "$combo"

    for numa in "${numa_nodes[@]}"; do

        TEST_DIR="$BASE_DIR/test_t${threads}s${streams}_n${numa}"
        mkdir -p "$TEST_DIR"

        echo "=== Running tests with ${threads} threads, ${streams} streams, NUMA $numa ==="

        for i in $(seq 1 $runs); do
            echo "Run #$i for t${threads}s${streams} on NUMA $numa"

            export RUN_ID=$i
            export EXPERIMENT_THREADS=$threads
            export EXPERIMENT_STREAMS=$streams
            export EXPERIMENT_NAME="local_t${threads}s${streams}_n${numa}_r${i}"
            export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"
            export THROUGHPUT_LOG_FILE="../../test_results/local_thread_numa_dependence/throughputs.txt"

            # Optional: Clear caches (if you want cold-start measurements)
            # sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"

            # Run the test pinned to the specified NUMA node(s)
            numactl -N "$numa" cmsRun "$local_script_name"
        done

        echo "Completed tests for threads=$threads, streams=$streams, NUMA=$numa"
    done
done

echo "All local (non-offload) tests completed!"
