#!/bin/bash
set -e
# =============================================================================
# L1 Track CI Run Unit Test
#
# Runs each algorithm in ALGOS on one dataset, all in parallel:
#   HYBRID, HYBRID_NEWKF -> ttbar+PU200 (prompt tracking)
#   HYBRID_DISPLACED -> displacedSUSY+PU200 (displaced tracking)
# Per algorithm: copy + edit cfg/plot macro/makeHists -> cmsRun -> makeHists.csh.
# Passes if every algorithm produces a non-empty results.out.
#
# COMMAND-LINE ARGS:
#   -e <num-events>     events per algorithm (default: 10)
#   -j <threads>        threads per cmsRun (default: 1); one cmsRun per algorithm
#                       in ALGOS runs at once, so total cores = (number of ALGOS) x <threads>
#
# HOW TO USE:
#   Input files are read by LFN. Run locally with a grid proxy until the
#   datasets are staged at the external (CI) site.
#   Registered in .../test/BuildFile.xml as L1TKRunTest (light: -e 10, for PRs).
#   For IB / heavy runs, set -e 1000 there and enable L1TKComparisonTest. Then from src/:
#       scram b runtests                  (all registered tests)
#       scram b runtests_L1TKRunTest      (this test only)
#   Outputs go to <working dir>/<ALGO>/, i.e. under SCRAM:
#       $CMSSW_BASE/unit_tests/<TestName>/<ALGO>/
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================
TEST_DIR="$CMSSW_BASE/src/L1Trigger/TrackFindingTracklet/test"
ORIG_CWD="$PWD"
# Algorithms to run (each in parallel). Remove entries to run a subset.
ALGOS=(HYBRID HYBRID_NEWKF HYBRID_DISPLACED)

EVENTS=10
JOBS=1
while getopts "e:j:" opt; do
    case "$opt" in
        e) EVENTS="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
    esac
done

# Input files (LFNs), written as Python list entries and pasted directly into
# cms.untracked.vstring(...) in the cfg. The shared directory is held in a
# variable, which bash fills in inside the double-quoted list.
# The lists cover at least 1000 events per dataset; cmsRun stops once maxEvents is reached.
#
# ttbar+PU200 checks prompt tracking (HYBRID, HYBRID_NEWKF).
# From /RelValTTbar_14TeV/CMSSW_20_0_0_pre1-PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3/GEN-SIM-DIGI-RAW
TTBAR_DIR="/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3/2590000"
MC_FILES_PROMPT="
    '$TTBAR_DIR/0438e4bc-b740-48a4-9d02-ff7896522eac.root',
    '$TTBAR_DIR/0658f9df-06ed-4f52-96c2-c3219a6e3063.root',
    '$TTBAR_DIR/0e0e3e92-ff7c-4366-b1bf-848139e9d85d.root',
    '$TTBAR_DIR/15b22169-65c6-4b4f-84fa-4e7fddb6c8fe.root',
    '$TTBAR_DIR/17e598ca-d193-490b-9ba2-b55799cb986b.root',
    '$TTBAR_DIR/1802417e-4c3c-43a8-bf85-f74cff1eec39.root',
    '$TTBAR_DIR/1e738605-01ec-46f7-b974-8874506e720d.root',
    '$TTBAR_DIR/226a8edf-ca8a-4304-9a4b-2014c3db908b.root',
    '$TTBAR_DIR/25298745-38aa-4b0f-8c08-a9fc1487a828.root',
    '$TTBAR_DIR/26ff480c-18d9-4ee7-b543-e6575118a7fb.root',
"

# displacedSUSY+PU200 checks displaced tracking (HYBRID_DISPLACED).
# From /RelValDisplacedSUSY_14TeV/CMSSW_20_0_0_pre1-PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3/GEN-SIM-DIGI-RAW
DISPLACED_DIR="/store/relval/CMSSW_20_0_0_pre1/RelValDisplacedSUSY_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3/2590000"
MC_FILES_DISPLACED="
    '$DISPLACED_DIR/05b92db6-4b4d-4dc6-a3a0-0450d986c033.root',
    '$DISPLACED_DIR/0667750b-443f-43de-b33e-15158545dca1.root',
    '$DISPLACED_DIR/06b029b5-98fb-44e8-bb5c-373260586f7f.root',
    '$DISPLACED_DIR/0884223d-099c-45de-91f3-813adf53abba.root',
    '$DISPLACED_DIR/0931e70c-736d-4c96-a4eb-6bd4e434b13d.root',
    '$DISPLACED_DIR/0a64d6f4-1837-4cf0-bd1d-0a991a6f030b.root',
    '$DISPLACED_DIR/0b7ce5e9-0f29-4817-91be-22598f424eca.root',
    '$DISPLACED_DIR/0c9e22c8-4c01-4c75-bdf9-426510b0da66.root',
    '$DISPLACED_DIR/0f27cde9-ee33-425e-8cab-152f08e3851c.root',
    '$DISPLACED_DIR/12845b9d-92ec-498b-8cb2-91e27e6edb20.root',
    '$DISPLACED_DIR/131292a7-cc9c-4954-a2e0-2d3489ef9da3.root',
    '$DISPLACED_DIR/1379584f-2424-4ea1-a17e-56c391e0c802.root',
    '$DISPLACED_DIR/13db5148-5e23-4b17-8c90-478bbf81da5d.root',
    '$DISPLACED_DIR/14ddab55-09ed-4722-a591-c3419924fd60.root',
    '$DISPLACED_DIR/160fc223-cdbc-4254-9fa0-8d1a0b4ad8fd.root',
    '$DISPLACED_DIR/1834d510-07b4-47fe-a8eb-33d966db311d.root',
    '$DISPLACED_DIR/196d7cb7-c2ce-498a-949a-6e388eb63970.root',
    '$DISPLACED_DIR/1b3e6956-36b2-4cc8-81db-5076fa9fdc5c.root',
    '$DISPLACED_DIR/1c3222a6-9700-445c-8a5d-17543ec20209.root',
    '$DISPLACED_DIR/1d281046-89e9-4c3b-a64a-55ac6f074ae6.root',
"

# =============================================================================
# HELPERS
# =============================================================================

# Copy a file, apply a sed -E edit, and fail loudly if the edit didn't land
# (e.g. if the line being replaced is ever renamed or reformatted upstream).
copy_and_edit() {   # <source> <destination> <sed-expression> <grep-check>
    cp "$1" "$2"
    sed -i -E "$3" "$2"
    grep -q "$4" "$2" || { echo "ERROR: edit '$3' did not apply to $2"; exit 1; }
}

# Stage 1a: give each algorithm its own edited copies, so parallel jobs never
# share a file and nothing is written under src/.
setup_job() {
    local ALGO=$1 DIR="$ORIG_CWD/$1"
    # cfg: select the algorithm
    copy_and_edit "$TEST_DIR/L1TrackNtupleMaker_cfg.py" "$DIR/job_cfg.py" \
        "s|L1TRKALGO = 'HYBRID'|L1TRKALGO = '$ALGO'|" "^L1TRKALGO = '$ALGO'"
    # Plot macro. Central's version has no useDisplacedTrkCuts switch, so for displaced
    # algorithms set the standard displaced settings (see the macro's own comment) in its
    # default arguments. Must keep its name -- ROOT calls the function matching the file name.
    if [[ "$ALGO" = *DISPLACED* ]]; then
        copy_and_edit "$TEST_DIR/L1TrackNtuplePlot.C" "$DIR/L1TrackNtuplePlot.C" \
            "s/(float TP_minPt = )[0-9.]+/\13.0/; s/(float TP_maxEta = )[0-9.]+/\12.0/; s/(float TP_maxLxy = )[0-9.]+/\110.0/; s/(float TP_maxD0 = )[0-9.]+/\110.0/; s/(bool doDetailedPlots = )(true|false)/\1true/" \
            "float TP_minPt = 3.0"
        for check in "float TP_minPt = 3.0" "float TP_maxEta = 2.0" "float TP_maxLxy = 10.0" \
                     "float TP_maxD0 = 10.0" "bool doDetailedPlots = true"; do
            grep -q "$check" "$DIR/L1TrackNtuplePlot.C" \
                || { echo "ERROR: '$check' not set in $DIR/L1TrackNtuplePlot.C"; exit 1; }
        done
    else
        cp "$TEST_DIR/L1TrackNtuplePlot.C" "$DIR/L1TrackNtuplePlot.C"   # prompt: keep the defaults
    fi
    # makeHists.csh: point the ROOT call at this job's macro copy
    copy_and_edit "$TEST_DIR/makeHists.csh" "$DIR/makeHists.csh" \
        "s|\\\$CMSSW_BASE/src/L1Trigger/TrackFindingTracklet/test/L1TrackNtuplePlot\.C|$DIR/L1TrackNtuplePlot.C|" \
        "$DIR/L1TrackNtuplePlot.C"

    # Append overrides to the END of job_cfg.py: the cfg runs top to bottom, so
    # these last assignments win over the defaults set earlier in the file.
    cat >> "$DIR/job_cfg.py" <<EOF
process.TFileService.fileName = cms.string('$DIR/histos.root')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32($EVENTS))
process.options.numberOfThreads = cms.untracked.uint32($JOBS)
process.options.numberOfStreams = cms.untracked.uint32(0)
EOF
}

# Stage 1b: point the cfg at this algorithm's input files, then run L1 tracking.
run_cmsrun() {
    local ALGO=$1 DIR="$ORIG_CWD/$1" FILES="$MC_FILES_PROMPT"
    [[ "$ALGO" = *DISPLACED* ]] && FILES="$MC_FILES_DISPLACED"
    echo "--- cmsRun: algo=$ALGO ---"

    echo "process.source.fileNames = cms.untracked.vstring($FILES)" >> "$DIR/job_cfg.py"

    cd "$DIR" && cmsRun job_cfg.py
    echo "--- cmsRun $ALGO: SUCCESS ---"
}

# Stage 2: make plots and results.out. tcsh/ROOT can exit 0 on failure,
# so check results.out explicitly.
run_makehists() {
    local DIR="$ORIG_CWD/$1"
    echo "--- makeHists: algo=$1 ---"
    cd "$DIR" && tcsh makeHists.csh histos.root
    [ -s results.out ] || { echo "FAILURE - results.out missing or empty for $1"; exit 1; }
    echo "--- makeHists $1: SUCCESS ---"
}

# One algorithm end to end. Kept as plain sequential lines (not &&) so that
# set -e still applies inside each step.
run_pipeline() {
    setup_job "$1"
    run_cmsrun "$1"
    run_makehists "$1"
}

# =============================================================================
# MAIN
# =============================================================================
# Compile quietly; print the build output only if it fails.
cd "$CMSSW_BASE/src"
if ! scram b -j > "$ORIG_CWD/compile.log" 2>&1; then
    echo "=== Compile: FAILURE -- log follows ==="
    cat "$ORIG_CWD/compile.log"
    exit 1
fi

# Output dirs must exist before launching, since each job's log is written there
for a in "${ALGOS[@]}"; do mkdir -p "$ORIG_CWD/$a"; done

declare -A PIDS
for a in "${ALGOS[@]}"; do
    run_pipeline "$a" > "$ORIG_CWD/$a/unit_test_run.log" 2>&1 &
    PIDS[$a]=$!
done

# Collect results. On success print one line per algorithm; on failure print
# that algorithm's full log, since on Jenkins only stdout/stderr is kept.
OVERALL_FAIL=0
for a in "${ALGOS[@]}"; do
    if wait "${PIDS[$a]}"; then
        echo "$a: SUCCESS"
    else
        OVERALL_FAIL=1
        echo ""
        echo "================ $a: FAILURE -- log follows ================"
        cat "$ORIG_CWD/$a/unit_test_run.log"
    fi
done

echo ""
if (( OVERALL_FAIL )); then
    echo "=== OVERALL RESULT: FAILURE ==="
    exit 1
fi
echo "=== OVERALL RESULT: SUCCESS ==="