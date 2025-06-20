#!/bin/bash

# Number of runs per test (first can be treated as warm-up)
runs=6

# Threads/Streams combinations to test
thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "24:24" "32:32")

# Message sizes in bytes
message_sizes=(4 1024 $((1024*1024))) # Example sizes: 1KB, 4KB, 16KB

# Script to run
script_local="dummy_configs/dummy_local_3send.py"
script_remote="dummy_configs/dummy_remote_3rec.py"

# Base directory for logs
BASE_DIR="../../test_results_thesis/dummy/mpich/async_number_of_products/different_machines"
mkdir -p "$BASE_DIR"

for message_size in "${message_sizes[@]}"; do
    export MESSAGE_SIZE=$message_size

    for combo in "${thread_stream_combos[@]}"; do
        IFS=':' read -r threads streams <<< "$combo"

        end_core=$((threads - 1))
        end_core_other=$((end_core + 64))
        TEST_DIR="$BASE_DIR/test_m${message_size}_t${threads}s${streams}"
        mkdir -p "$TEST_DIR"

        echo "=== Running tests with ${threads} threads, ${streams} streams, message size ${message_size} bytes, CPUs: 0-$end_core ==="

        for i in $(seq 1 $runs); do
            echo "Run #$i for m${message_size}_t${threads}s${streams} on CPU list: 0-$end_core"

            export RUN_ID=$i
            export EXPERIMENT_THREADS=$threads
            export EXPERIMENT_STREAMS=$streams
            export EXPERIMENT_NAME="xpmem_m${message_size}_t${threads}s${streams}_r${i}"
            export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"
            export THROUGHPUT_LOG_FILE="$BASE_DIR/throughputs.txt"

            # Run pinned to the CPU list
            /nfshome0/apolova/mpich-4.3.0-install/bin/mpirun -np 1 env \
                MPIR_CVAR_CH4_DEVICE=ch4:ucx \
                UCX_TLS=xpmem,self,shm \
                MPIR_CVAR_CH4_NUM_VCIS=$threads \
                MPIR_CVAR_CH4_UCX_USE_MULTIPLE_EP=1 \
                MPIR_CVAR_CH4_VCI_METHOD=per_vci \
                MESSAGE_SIZE=$message_size \
                numactl --physcpubind=64-"${end_core_other}" cmsRun "$script_remote" \
                : -np 1 env \
                MPIR_CVAR_CH4_DEVICE=ch4:ucx \
                UCX_TLS=xpmem,self,shm \
                MESSAGE_SIZE=$message_size \
                MPIR_CVAR_CH4_NUM_VCIS=$threads \
                MPIR_CVAR_CH4_UCX_USE_MULTIPLE_EP=1 \
                MPIR_CVAR_CH4_VCI_METHOD=per_vci \
                numactl --physcpubind=0-"${end_core}" cmsRun "$script_local"

        done

        echo "Completed tests for threads=$threads, streams=$streams, message size=$message_size bytes"
    done
done

echo "All local (non-offload) tests completed!"
