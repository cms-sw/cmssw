#!/usr/bin/env bash
#
# Build a DOT/SVG gallery of logical truth graphs for the enableTruth relval
# samples, one folder per physics process. For each sample it dumps:
#   <label>_full_eventN.dot      - the full logical graph (seedPdgIds=0, --showAll)
#   <label>_selected_eventN.dot  - the natural-seed selection (+ rendered .svg)
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

# label : workflow-number : natural seed PDG ids for the selection
SAMPLES=(
  "SingleElectron:34002:11,-11"
  "TTbar:34034:6,-6"
  "DYToLL:34044:23"
  "DYToTauTau:34045:23"
  "ZMM:34050:23"
  "H125_diphoton:34052:25"
  "VBFHZZ4Nu:34131:25"
  "TenTau:34087:15,-15"
)

rm -rf "$OUT"; mkdir -p "$OUT"
cmds=()
for s in "${SAMPLES[@]}"; do
  IFS=: read -r lab num seeds <<< "$s"
  step3=$(ls "$LIB"/${num}.88_*/step3.root 2>/dev/null | head -1)
  if [[ -z "$step3" ]]; then echo "SKIP $lab ($num): no step3.root under $LIB"; continue; fi
  mkdir -p "$OUT/$lab"
  cmds+=("cmsRun $CFG file:$step3 -n $NEVT -o $OUT/$lab -s $seeds -d 1 -t _sel > $OUT/$lab/sel.log 2>&1")
  cmds+=("cmsRun $CFG file:$step3 -n 1 --showAll -s 0 -t _full -o $OUT/$lab > $OUT/$lab/full.log 2>&1")
done

[[ ${#cmds[@]} -eq 0 ]] && { echo "No samples found under $LIB"; exit 1; }
printf '%s\n' "${cmds[@]}" | xargs -d '\n' -P "$JOBS" -I CMD bash -c 'CMD'

# keep only logical graphs, rename by real event id, render selected views to SVG
for d in "$OUT"/*/; do
  lab=$(basename "$d"); pushd "$d" >/dev/null
  rm -f truthgraph_*.dot ./*.log rechits_nano*.root
  for f in truthlogicalgraph_sel_run1_lumi1_event*.dot; do
    [[ -f "$f" ]] || continue; ev=$(grep -o 'event[0-9]*' <<< "$f"); mv -f "$f" "${lab}_selected_${ev}.dot"
  done
  for f in truthlogicalgraph_full_run1_lumi1_event*.dot; do
    [[ -f "$f" ]] || continue; ev=$(grep -o 'event[0-9]*' <<< "$f"); mv -f "$f" "${lab}_full_${ev}.dot"
  done
  popd >/dev/null
done
ls "$OUT"/*/*_selected_event*.dot 2>/dev/null | \
  xargs -P "$JOBS" -I {} bash -c 'timeout 300 dot -Tsvg "{}" -o "${1%.dot}.svg" 2>/dev/null' _ {}

echo "Gallery written to $OUT"
for d in "$OUT"/*/; do
  echo "  $(basename "$d"): $(ls "$d"/*.dot 2>/dev/null | wc -l) dot, $(ls "$d"/*.svg 2>/dev/null | wc -l) svg"
done
