#!/bin/bash

# Number of runs per test (first can be treated as warm-up)
runs=6

# Threads/Streams combinations to test
thread_stream_combos=("1:1" "4:4" "8:8" "16:16" "24:24" "32:32")

# Script to run
script_local="hlt_local.py"
script_remote="hlt_remote.py"

# Base directory for logs
BASE_DIR="../../test_results_thesis/mpich/async_number_of_products/local-remote_t-s-c_different-sockets"
mkdir -p "$BASE_DIR"


for combo in "${thread_stream_combos[@]}"; do
    IFS=':' read -r threads streams <<< "$combo"

    end_core=$((threads - 1))
    end_core_other=$((end_core + 64))
    TEST_DIR="$BASE_DIR/test_t${threads}s${streams}"
    mkdir -p "$TEST_DIR"

    echo "=== Running tests with ${threads} threads, ${streams} streams, CPUs: 0-$end_core ==="

    for i in $(seq 1 $runs); do
        echo "Run #$i for t${threads}s${streams} on CPU list: 0-$end_core"

        export RUN_ID=$i
        export EXPERIMENT_THREADS=$threads
        export EXPERIMENT_STREAMS=$streams
        export EXPERIMENT_NAME="xpmem_t${threads}s${streams}_r${i}"
        export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"
        export THROUGHPUT_LOG_FILE="$BASE_DIR/throughputs.txt"
        # export OMPI_MCA_pml=ob1
        # export OMPI_MCA_btl_vader_single_copy_mechanism=xpmem
        # export OMPI_MCA_btl=self,vader

        # Run pinned to the CPU list
        /nfshome0/apolova/mpich-4.3.0-install/bin/mpirun -np 1 env MPIR_CVAR_CH4_DEVICE=ch4:ucx UCX_TLS=xpmem,self,shm numactl --physcpubind=64-"${end_core_other}" cmsRun "$script_remote" \
              : -np 1 env MPIR_CVAR_CH4_DEVICE=ch4:ucx UCX_TLS=xpmem,self,shm numactl --physcpubind=0-"${end_core}" cmsRun "$script_local"
    done

    echo "Completed tests for threads=$threads, streams=$streams"
done

echo "All local (non-offload) tests completed!"
