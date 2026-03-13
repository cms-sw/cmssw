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

    if [[ ! -e "$cfg" ]]; then
        echo "Config $cfg not found"
        return
    fi

    ensure_patatrack_scripts

    patatrack-scripts/benchmark \
        -j 8 -t 16 -s 16 \
        -e ${EVENTS} \
        --no-input-benchmark \
        --slot "numa=0-3:mem=0-3" \
        --event-skip 100 \
        --event-resolution 10 \
        -k Phase2Timing_resources.json \
        -- ${cfg}

    check_logs_for_errors || exit 1

    mergeResourcesJson.py logs/step*/pid*/Phase2Timing_resources.json > ${output_json}
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
