#!/usr/bin/env bash
#
# Run the enableTruth Run4 D120 (no-PU) truth-validation relval sample set.
# These are the workflows used to exercise the truth graph across topologies:
#   34002 SingleElectronPt35   34034 TTbar_14TeV        34044 DYToLL_M-50
#   34045 DYToTauTau_M-50       34050 ZMM_14TeV          34052 H125 ggF
#   34131 VBFHZZ4Nu             34087 TenTau (E 15-500)
# The .88 offset is the enableTruth UpgradeWorkflow variant; with the committed
# fix its GenSim step also carries enableTruth (PersistencyEmin=0) so the truth
# graph is connected.
#
# Requires cmsenv. Usage:
#   cmsenv
#   runTruthRelvals.sh [OUTPUT_DIR]      (default ./library)
# Env knobs: JOBS (parallel workflows, default 8), THREADS (per job, default 8),
#            WF (override the comma-separated workflow list).
#
set -uo pipefail
: "${CMSSW_BASE:?run cmsenv first}"

OUT="${1:-$PWD/library}"
WF="${WF:-34002.88,34034.88,34044.88,34045.88,34050.88,34052.88,34131.88,34087.88}"
JOBS="${JOBS:-8}"
THREADS="${THREADS:-8}"

mkdir -p "$OUT"
cd "$OUT"
echo "Running workflows [$WF] into $OUT (jobs=$JOBS threads=$THREADS)"
runTheMatrix.py -w upgrade -l "$WF" -j "$JOBS" -t "$THREADS"
