#!/usr/bin/env bash
#
# Build a DOT/SVG gallery of logical truth graphs for the enableTruth relval
# samples, one folder per physics process. For each sample it dumps:
#   <label>_full_eventN.dot      - the full logical graph (seedPdgIds=0, --showAll)
#   <label>_signal_eventN.dot    - the natural-seed signal view (+ rendered .svg):
#                                  seedParentDepth=0, so the truncated upstream of
#                                  each seed is summarized into one per-interaction
#                                  Interaction -> {ISR/upstream, UnderlyingEvent}
#                                  artificial-source structure (the signal-vs-rest
#                                  split). Everything reachable from the signal
#                                  Interaction vertex is, by definition, the signal.
#
# Requires cmsenv. Usage:
#   cmsenv
#   makeTruthGallery.sh [LIBRARY_DIR] [OUTPUT_DIR]
#     LIBRARY_DIR  dir containing <wf>.88_*/step3.root   (default ./library)
#     OUTPUT_DIR   gallery output dir                    (default ./dot_gallery)
# Env knobs: NEVT (selected events/sample, default 3), JOBS (parallel, default 16).
#
set -uo pipefail
: "${CMSSW_BASE:?run cmsenv first}"

LIB="${1:-$PWD/library}"
OUT="${2:-$PWD/dot_gallery}"
CFG="$CMSSW_BASE/src/PhysicsTools/TruthInfo/test/dumpTruthGraphsFromGENSIMRECO_cfg.py"
NEVT="${NEVT:-3}"
JOBS="${JOBS:-16}"

# label : workflow-number : natural seed PDG ids : optional extra dump flags.
# --keepProductionSiblings is added for VBF, whose seed (the Higgs) recoils against
# the two tagging quarks at its production vertex: the signal view then shows the
# real hard vertex and the forward jets rather than summarizing them into the
# artificial Upstream node. (It is a no-op for 2->1 production like ggF gg->H or
# s-channel Drell-Yan, where the seed has no production-vertex co-products.)
SAMPLES=(
  "SingleElectron:34002:11,-11:"
  "TTbar:34034:6,-6:"
  "DYToLL:34044:23:"
  "DYToTauTau:34045:23:"
  "ZMM:34050:23:"
  "H125_diphoton:34052:25:"
  "VBFHZZ4Nu:34131:25:--keepProductionSiblings"
  "TenTau:34087:15,-15:"
)

rm -rf "$OUT"; mkdir -p "$OUT"
cmds=()
for s in "${SAMPLES[@]}"; do
  IFS=: read -r lab num seeds extra <<< "$s"
  step3=$(ls "$LIB"/${num}.88_*/step3.root 2>/dev/null | head -1)
  if [[ -z "$step3" ]]; then echo "SKIP $lab ($num): no step3.root under $LIB"; continue; fi
  mkdir -p "$OUT/$lab"
  cmds+=("cmsRun $CFG file:$step3 -n $NEVT -o $OUT/$lab -s $seeds -d 0 $extra -t _sig > $OUT/$lab/sig.log 2>&1")
  cmds+=("cmsRun $CFG file:$step3 -n 1 --showAll -s 0 -t _full -o $OUT/$lab > $OUT/$lab/full.log 2>&1")
done

[[ ${#cmds[@]} -eq 0 ]] && { echo "No samples found under $LIB"; exit 1; }
printf '%s\n' "${cmds[@]}" | xargs -d '\n' -P "$JOBS" -I CMD bash -c 'CMD'

# keep only logical graphs, rename by real event id, render signal views to SVG
for d in "$OUT"/*/; do
  lab=$(basename "$d"); pushd "$d" >/dev/null
  rm -f truthgraph_*.dot ./*.log rechits_nano*.root
  for f in truthlogicalgraph_sig_run1_lumi1_event*.dot; do
    [[ -f "$f" ]] || continue; ev=$(grep -o 'event[0-9]*' <<< "$f"); mv -f "$f" "${lab}_signal_${ev}.dot"
  done
  for f in truthlogicalgraph_full_run1_lumi1_event*.dot; do
    [[ -f "$f" ]] || continue; ev=$(grep -o 'event[0-9]*' <<< "$f"); mv -f "$f" "${lab}_full_${ev}.dot"
  done
  popd >/dev/null
done
# Large signal graphs (full SIM cascade, O(10^4) nodes) need a generous layout
# budget; the dense gun samples (TenTau) take several minutes in dot.
ls "$OUT"/*/*_signal_event*.dot 2>/dev/null | \
  xargs -P "$JOBS" -I {} bash -c 'timeout 1500 dot -Tsvg "{}" -o "${1%.dot}.svg" 2>/dev/null' _ {}

echo "Gallery written to $OUT"
for d in "$OUT"/*/; do
  echo "  $(basename "$d"): $(ls "$d"/*.dot 2>/dev/null | wc -l) dot, $(ls "$d"/*.svg 2>/dev/null | wc -l) svg"
done
