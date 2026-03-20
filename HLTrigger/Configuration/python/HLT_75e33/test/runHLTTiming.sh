#!/usr/bin/env bash

# Exit the script immediately if any command fails
set -e

# Enable pipefail to propagate the exit status of the entire pipeline
set -o pipefail

############################
# Global configuration
############################

FOLDER_FILES="/data/user/${USER}/"
DATASET="/RelValTTbar_14TeV/CMSSW_15_1_0_pre3-PU_150X_mcRun4_realistic_v1_STD_Run4D110_PU-v1/GEN-SIM-DIGI-RAW"

EVENTS=1000
THREADS=4

############################
# GPU Monitoring config
############################

ENABLE_GPU_MONITORING=true
MONITOR_INTERVAL=1

# Check dependencies
if [[ "$ENABLE_GPU_MONITORING" = true ]]; then
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: nvidia-smi not found but GPU monitoring enabled"
        exit 1
    fi
fi

############################
# Utility functions
############################

check_logs_for_errors() {
    local log_dirs=${1:-"logs/step*/pid*"}
    local error_found=0

    for f in $log_dirs/stdout $log_dirs/stderr; do
        if [[ -f "$f" ]]; then
            if grep -qiE 'error|fail|exception|traceback' "$f"; then
                echo "Error keyword found in: $f"
                error_found=1
            fi
        fi
    done

    if [[ $error_found -eq 1 ]]; then
        echo "Failure detected in logs."
        return 1
    fi
}

ensure_patatrack_scripts() {
    if [[ ! -d patatrack-scripts ]]; then
        git clone https://github.com/cms-patatrack/patatrack-scripts --depth 1
    fi
}

get_current_total_gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk '{ total += $1 } END { print total }'
}

get_current_gpus_usage() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        | paste -sd ','
}

############################
# Data handling
############################

fetch_files() {

    mapfile -t FILES < <(
        dasgoclient -query="file dataset=${DATASET}" --limit=-1 |
        sort |
        head -4
    )

    for f in "${FILES[@]}"; do

        local mypath
        mypath=$(dirname "$f")

        mkdir -p "${FOLDER_FILES}${mypath}"

        if [[ -e "/eos/cms/$f" && ! -e "${FOLDER_FILES}${f}" ]]; then
            echo "Copying $f"
            cp "/eos/cms/$f" "${FOLDER_FILES}${mypath}"
        fi
    done
}

build_input_file_string() {

    LOCALPATH=${FOLDER_FILES}$(dirname ${FILES[0]})

    echo "Local repository: |${LOCALPATH}|"

    ALL_FILES=""

    for f in $(ls -1 ${LOCALPATH}); do
        ALL_FILES+="file:${LOCALPATH}/${f},"
    done

    ALL_FILES="${ALL_FILES%?}"

    echo "Discovered files: $ALL_FILES"
}

############################
# cmsDriver generator
############################

run_cmsdriver() {

    local fragment=$1
    local menu=$2
    local process=$3
    local output_py=$4
    local extra_args=$5

    cmsDriver.py ${fragment} \
        -s ${menu} \
        --processName=${process} \
        --conditions auto:phase2_realistic_T35 \
        --geometry ExtendedRun4D110 \
        --era Phase2C17I13M9 \
        --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 \
        --eventcontent FEVTDEBUGHLT \
        --filein="${ALL_FILES}" \
        --mc \
        --nThreads ${THREADS} \
        --inputCommands 'keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
        -n ${EVENTS} \
        --no_exec \
        --output {} \
        ${extra_args} \
        --python_filename ${output_py}
}

############################
# Benchmark runner
############################

run_benchmark() {

    local cfg=$1
    local output_json=$2
    local logdir="logs.$(basename ${cfg%.py})"

    if [[ ! -e "$cfg" ]]; then
        echo "Config $cfg not found"
        return
    fi

    ensure_patatrack_scripts
    mkdir -p "$logdir"

    if [[ "$ENABLE_GPU_MONITORING" = true ]]; then

        echo "Running benchmark WITH GPU monitoring"

        local CSV_FILE="${logdir}/gpu_memory.csv"
        local CSV_GPU_FILE="${logdir}/gpu_usage.csv"
        local TMP_LOG_FILE="${logdir}/benchmark.tmp.log"

        echo "elapsed_seconds,memory_mib" > "$CSV_FILE"
        echo "elapsed_seconds,gpu_usage" > "$CSV_GPU_FILE"

        local max_mem=0
        local sum_mem=0
        local count=0

        declare -a totals
        declare -a max_usage

        local START_TIME=$(date +%s)

        # Run benchmark in background
        patatrack-scripts/benchmark \
            -j 8 -t 16 -s 16 \
            -e ${EVENTS} \
            --no-input-benchmark \
            --slot "numa=0-3:mem=0-3" \
            --event-skip 100 \
            --event-resolution 10 \
            -k Phase2Timing_resources.json \
            -- ${cfg} > "$TMP_LOG_FILE" 2>&1 &

        local PID=$!

        # Live output
        tail -f "$TMP_LOG_FILE" &
        local TAIL_PID=$!

        trap "kill $PID $TAIL_PID 2>/dev/null" EXIT

        while ps -p $PID > /dev/null; do

            # Memory
            mem=$(get_current_total_gpu_mem)
            now=$(date +%s)
            elapsed=$((now - START_TIME))

            if [[ "$mem" =~ ^[0-9]+$ ]]; then
                echo "$elapsed,$mem" >> "$CSV_FILE"
                ((mem > max_mem)) && max_mem=$mem
                sum_mem=$((sum_mem + mem))
                count=$((count + 1))
            fi

            # GPU usage
            usage=$(get_current_gpus_usage)
            if [[ "$usage" =~ ^[0-9,]+$ ]]; then
                echo "$elapsed,$usage" >> "$CSV_GPU_FILE"

                IFS=',' read -ra vals <<< "$usage"
                for i in "${!vals[@]}"; do
                    totals[$i]=$((${totals[$i]:-0} + vals[$i]))
                    ((vals[$i] > ${max_usage[$i]:-0})) && max_usage[$i]=${vals[$i]}
                done
            fi

            sleep $MONITOR_INTERVAL
        done

        kill $TAIL_PID 2>/dev/null
        mv "$TMP_LOG_FILE" "${logdir}/output.log"

        # Compute mean
        if ((count > 0)); then
            mean_mem=$((sum_mem / count))
        else
            mean_mem=0
        fi

        {
            echo ""
            echo "----- GPU SUMMARY -----"
            echo "Peak memory: ${max_mem} MiB"
            echo "Mean memory: ${mean_mem} MiB"
            echo ""
            echo "Per-GPU usage:"
            for i in "${!totals[@]}"; do
                avg=$((totals[$i] / count))
                echo "GPU $i: avg=${avg}% max=${max_usage[$i]}%"
            done
            echo "-----------------------"
        } | tee -a "${logdir}/output.log"

    else

        echo "Running benchmark WITHOUT GPU monitoring"

        patatrack-scripts/benchmark \
            -j 8 -t 16 -s 16 \
            -e ${EVENTS} \
            --no-input-benchmark \
            --slot "numa=0-3:mem=0-3" \
            --event-skip 100 \
            --event-resolution 10 \
            -k Phase2Timing_resources.json \
            -- ${cfg} | tee "${logdir}/output.log"
    fi

    check_logs_for_errors || exit 1

    mergeResourcesJson.py logs/step*/pid*/Phase2Timing_resources.json > "${output_json}"
}

############################
# Workflows
############################

run_phase2_gpu() {

    run_cmsdriver \
        "Phase2" \
        "L1P2GT,HLT:75e33_timing" \
        "HLTX" \
        "Phase2_L1P2GT_HLT.py" \
        ""

    run_benchmark \
        "Phase2_L1P2GT_HLT.py" \
        "Phase2Timing_resources.json"

    if [[ -e "$(dirname $0)/augmentResources.py" ]]; then
        python3 $(dirname $0)/augmentResources.py
    fi
}

run_phase2_cpu() {

    run_cmsdriver \
        "Phase2" \
        "L1P2GT,HLT:75e33_timing" \
        "HLTX" \
        "Phase2_L1P2GT_HLT_OnCPU.py" \
        "--accelerators cpu"

    run_benchmark \
        "Phase2_L1P2GT_HLT_OnCPU.py" \
        "Phase2Timing_resources_OnCPU.json"
}

run_ngt_scouting() {

    run_cmsdriver \
        "NGTScouting" \
        "L1P2GT,HLT:NGTScouting" \
        "NLTX" \
        "NGTScouting_L1P2GT_HLT.py" \
        "--procModifiers ngtScouting"

    run_benchmark \
        "NGTScouting_L1P2GT_HLT.py" \
        "Phase2Timing_resources_NGT.json"
}

############################
# Main
############################

main() {

    fetch_files
    build_input_file_string

    run_phase2_gpu
    run_phase2_cpu
    run_ngt_scouting
}

main "$@"
