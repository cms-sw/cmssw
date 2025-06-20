#!/bin/bash

# MPICH/UCX parameters to vary
vcis_values=(1 4 8 16 32)
reserved_vcis=0  # can be changed if needed

# Number of runs per test
runs=5  # 4 real + 1 warmup

# Threads/Streams combinations
thread_stream_combos=("4:4" "8:8" "16:16" "32:32" "64:64" "128:128" "256:256")

# NUMA configuration
local_numa=6
remote_numa=7

# Scripts to run
local_script_name="hlt_local.py"
remote_script_name="hlt_remote.py"

# MPICH installation path
MPIRUN="/nfshome0/apolova/mpich-4.3.0-install/bin/mpirun"
export LD_LIBRARY_PATH=/nfshome0/apolova/ucx-1.18.1/lib:$LD_LIBRARY_PATH

# Output dir
BASE_DIR="../../test_results/sync_mpich_vci_dependence"
mkdir -p "$BASE_DIR"

for vcis in "${vcis_values[@]}"; do
    for combo in "${thread_stream_combos[@]}"; do
        IFS=':' read -r threads streams <<< "$combo"
        TEST_DIR="$BASE_DIR/test_vci${vcis}_t${threads}s${streams}_n${local_numa}-${remote_numa}"
        mkdir -p "$TEST_DIR"

        echo "=== Testing VCIs=$vcis with t=${threads}, s=${streams}, NUMA $local_numa:$remote_numa ==="

        for i in $(seq 1 $runs); do
            echo "--- Run #$i ---"

            export RUN_ID=$i
            export MPIR_CVAR_CH4_DEVICE=ch4:ucx
            export MPIR_CVAR_CH4_NUM_VCIS=$vcis
            export MPIR_CVAR_CH4_RESERVE_VCIS=$reserved_vcis
            export MPIR_CVAR_CH4_ROOTS_ONLY=1
            export MPICH_ENV_DISPLAY=1
            export UCX_TLS=xpmem,self,shm
            export UCX_LOG_LEVEL=warn

            export EXPERIMENT_THREADS=$threads
            export EXPERIMENT_STREAMS=$streams
            export EXPERIMENT_NAME="vci${vcis}_t${threads}s${streams}_n${local_numa}-${remote_numa}_r${i}"
            export EXPERIMENT_OUTPUT_DIR="$TEST_DIR"

            local_log="$TEST_DIR/local_run_$i.log"
            remote_log="$TEST_DIR/remote_run_$i.log"
            local_thr="$TEST_DIR/local_throughput_$i.txt"
            remote_thr="$TEST_DIR/remote_throughput_$i.txt"

            # Run with mpich mpirun
            $MPIRUN -np 1 env THROUGHPUT_LOG_FILE="$remote_thr" numactl -N $remote_numa cmsRun $remote_script_name \
              : -np 1 env THROUGHPUT_LOG_FILE="$local_thr" numactl -N $local_numa cmsRun $local_script_name \
              > >(tee "$TEST_DIR/run_$i.out") 2>&1

            if [[ $i -eq 1 ]]; then
                echo "Discarding warmup run $i"
                rm -f "$local_log" "$remote_log" "$local_thr" "$remote_thr"
            fi
        done
    done
done

echo "=== All MPICH VCI scaling tests completed ==="
