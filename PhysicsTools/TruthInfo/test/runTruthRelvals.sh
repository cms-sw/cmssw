#!/usr/bin/env bash
#
# Run the Run4 D127 (no-PU) truth-validation relval sample set, one workflow per truth
# topology. Truth needs no workflow variant: every Run4 era carries enableTruth, so the
# plain .0 workflow builds the graph, runs the associators and fills the truth DQM.
#   37602 SingleElectronPt35  37605 SingleGammaPt35   37607 SingleMuPt10
#   37688 SinglePiPt25        37687 TenTau E15to500   37634 TTbar_14TeV
#   37644 DYToLL_M-50         37645 DYToTauTau_M-50   37646 ZEE_14
#   37650 ZMM_14              37652 H125 ggF          37731 VBFHZZ4Nu
#   37640 MinBias_14          37643 QCDForPF_14
# The vh, singletop and diboson presets have no relval workflow; produce them with the
# custom fragments in PhysicsTools/TruthInfo/python.
# ZMM: four of its events stop RECO in TICLCandidateProducer, a release defect, so add
# process.options.TryToContinue = cms.untracked.vstring("BadAlloc") to skip them.
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
WF="${WF:-37602.0,37605.0,37607.0,37688.0,37687.0,37634.0,37644.0,37645.0,37646.0,37650.0,37652.0,37731.0,37640.0,37643.0}"
JOBS="${JOBS:-8}"
THREADS="${THREADS:-8}"

mkdir -p "$OUT"
cd "$OUT"
echo "Running workflows [$WF] into $OUT (jobs=$JOBS threads=$THREADS)"
runTheMatrix.py -w upgrade -l "$WF" -j "$JOBS" -t "$THREADS"
