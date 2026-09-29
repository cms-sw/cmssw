#!/usr/bin/env bash
#
# Render the truth-Branch DQM validation plots for the enableTruth relval
# library, overlaying a few representative samples in one set of PNGs+index.html.
# It feeds the per-sample harvested legacy DQM file produced by step4 of each
# workflow (DQM_V0001_R*__Global__*__RECO.root) into makeTruthGraphValidationPlots.py,
# which derives the efficiency / fake-rate / self-match ratios and overlays the
# booked quality distributions (purity, completeness, response, n_sharing_branches,
# best-matched-Branch metrics).
#
# Requires cmsenv. Usage:
#   cmsenv
#   makeBranchValidationPlots.sh [LIBRARY_DIR] [OUTPUT_DIR]
#     LIBRARY_DIR  dir containing <wf>.88_*/DQM_V0001_*__RECO.root  (default ./library)
#     OUTPUT_DIR   plots output dir                                 (default ./branch_validation_plots)
# Env knobs: SAMPLES (override the "label:workflow" overlay list).
#
set -uo pipefail
: "${CMSSW_BASE:?run cmsenv first}"

LIB="${1:-$PWD/library}"
OUT="${2:-$PWD/branch_validation_plots}"
PLOTTER="$CMSSW_BASE/src/PhysicsTools/TruthInfo/scripts/makeTruthGraphValidationPlots.py"

# label : workflow-number. Diverse topologies: hadronic (TTbar), dense multi-tau
# gun (TenTau, where the calo n_sharing tail is richest) and clean dimuon (ZMM).
SAMPLES_DEFAULT=("TTbar:34034" "TenTau:34087" "ZMM:34050")
read -r -a SAMPLES <<< "${SAMPLES:-${SAMPLES_DEFAULT[*]}}"

args=()
for s in "${SAMPLES[@]}"; do
  IFS=: read -r lab num <<< "$s"
  dqm=$(ls "$LIB"/${num}.88_*/DQM_V0001_R*__Global__*__RECO.root 2>/dev/null | head -1)
  if [[ -z "$dqm" ]]; then echo "SKIP $lab ($num): no harvested DQM under $LIB"; continue; fi
  args+=("${dqm}:${lab}")
done

[[ ${#args[@]} -eq 0 ]] && { echo "No harvested DQM files found under $LIB"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
echo "Plotting [${args[*]##*:}] into $OUT"
python3 "$PLOTTER" "${args[@]}" -o "$OUT"
echo "Branch validation plots written to $OUT (open $OUT/index.html)"
