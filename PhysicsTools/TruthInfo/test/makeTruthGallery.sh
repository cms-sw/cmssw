#!/usr/bin/env bash
#
# Build a DOT/SVG gallery of logical truth graphs for the enableTruth relval
# samples, one folder per physics process. For each sample it dumps:
#   <label>_full_eventN.dot      - the full logical graph (seedPdgIds=0)
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
#     LIBRARY_DIR  dir containing <wf>.0_*/step3.root    (default ./library)
#     OUTPUT_DIR   gallery output dir                    (default ./dot_gallery)
# Env knobs: NEVT (selected events/sample, default 3), JOBS (parallel, default 16).
#
set -uo pipefail
: "${CMSSW_BASE:?run cmsenv first}"

LIB="${1:-$PWD/library}"
OUT="${2:-$PWD/dot_gallery}"
CFG="$CMSSW_BASE/src/PhysicsTools/TruthInfo/test/dumpTruthGraphsFromGENSIMRECO_cfg.py"
SELECT="$CMSSW_BASE/src/PhysicsTools/TruthInfo/python/truthGraphSelections.py"
# DOT layout: 'dot' (default, hierarchical left-to-right ranks) or a force-directed
# engine ('sfdp'/'fdp'/'neato') for node repulsion + spring edges toward parents.
# ENGINE is the graphviz binary used to render and follows LAYOUT by default, so
# e.g. `LAYOUT=sfdp makeTruthGallery.sh ...` produces the force-directed view.
LAYOUT="${LAYOUT:-dot}"
ENGINE="${ENGINE:-$LAYOUT}"
NEVT="${NEVT:-3}"
# The dumper reads hit and vertex positions through the geometry, so it must be the one
# the sample was produced with.
GEOMETRY="${GEOMETRY:-ExtendedRun4D127}"
JOBS="${JOBS:-16}"

# label : workflow-number. The per-process signal-view selection (seed, parent
# depth, decay channel, --keepProductionSiblings, ...) is no longer hard-coded:
# it is resolved from each workflow's generator fragment by truthGraphSelections.py
# (the seven enableTruth presets), so VBF gets the tagging quarks, Drell-Yan gets
# the dilepton channel, guns get their species, etc., and a new sample needs no
# edit here. Override per sample by appending flags after $flags below if needed.
SAMPLES=(
  "SingleElectron:37602"
  "SingleGamma:37605"
  "SingleMu:37607"
  "SinglePi:37688"
  "TTbar:37634"
  "DYToLL:37644"
  "DYToTauTau:37645"
  "ZEE:37646"
  "ZMM:37650"
  "H125_diphoton:37652"
  "VBFHZZ4Nu:37731"
  "TenTau:37687"
  "MinBias:37640"
  "QCDForPF:37643"
  # No single-top sample exists in the relval matrix; 34999 is a custom ST t-channel
  # (PhysicsTools/TruthInfo/ST_tch_top_14TeV_TuneCP5_cfi) produced locally to exercise
  # the 'singletop' preset. The three custom numbers below are local, not matrix
  # workflows, so the gallery skips them unless the library holds them.
  "SingleTop:34999"
  # 34998.88 is a custom ttbar-POWHEG (TTto2L2Nu, 13.6 TeV gridpack -> 14 TeV Phase-2,
  # see the WARNING in PhysicsTools/TruthInfo/TTto2L2Nu_Powheg_Pythia8_cfi): an NLO
  # ttbar example alongside the LO Pythia8 ttbar (37634).
  "TTbarPowheg:34998"
  # 34997.88 is a custom diboson WW->2l2nu (PhysicsTools/TruthInfo/WWTo2L2Nu_14TeV_TuneCP5_cfi):
  # exercises the 'diboson' preset (seed the vector bosons {23,24,-24} + production system).
  "Diboson:34997"
  # 34996.88 is a custom VH ZH->bb,ll (PhysicsTools/TruthInfo/ZHToBB_ZToLL_14TeV_TuneCP5_cfi):
  # exercises the 'vh' preset (seed the Higgs {25}, keep the recoiling Z as a sibling).
  "VH:34996"
)

rm -rf "$OUT"; mkdir -p "$OUT"
cmds=()
for s in "${SAMPLES[@]}"; do
  IFS=: read -r lab num <<< "$s"
  step3=$(ls "$LIB"/${num}.[08]*_*/step3.root 2>/dev/null | head -1)
  if [[ -z "$step3" ]]; then echo "SKIP $lab ($num): no step3.root under $LIB"; continue; fi
  # Generator fragment from the workflow dir name (strip the "<wf>.<variant>_" prefix and
  # the "+Run4..." suffix) -> per-process selection preset.
  frag=$(basename "$(dirname "$step3")"); frag=${frag#*_}; frag=${frag%%+*}
  flags=$(python3 "$SELECT" "$frag")
  echo "  $lab: fragment '$frag' -> $(python3 "$SELECT" "$frag" --name) [$flags]"
  mkdir -p "$OUT/$lab"
  cmds+=("cmsRun $CFG file:$step3 -n $NEVT -o $OUT/$lab $flags --geometry $GEOMETRY --layout $LAYOUT -t _sig > $OUT/$lab/sig.log 2>&1")
  cmds+=("cmsRun $CFG file:$step3 -n 1 -s 0 -d 0 --geometry $GEOMETRY --layout $LAYOUT -t _full -o $OUT/$lab > $OUT/$lab/full.log 2>&1")
done

[[ ${#cmds[@]} -eq 0 ]] && { echo "No samples found under $LIB"; exit 1; }
printf '%s\n' "${cmds[@]}" | xargs -d '\n' -P "$JOBS" -I CMD bash -c 'CMD'

# Publish the rechit NanoAOD tables (recHitTable + PFRecHitFlatTable +
# TrackerSimHitFlatTableProducer) the dump produces, one per sample, in a separate
# rechits/ folder; then keep only the logical graphs, rename by real event id.
mkdir -p "$OUT/rechits"
for d in "$OUT"/*/; do
  lab=$(basename "$d"); [[ "$lab" == rechits ]] && continue
  pushd "$d" >/dev/null
  [[ -f rechits_nano_sig.root ]] && mv -f rechits_nano_sig.root "$OUT/rechits/${lab}_rechits.root"
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
export ENGINE
ls "$OUT"/*/*_signal_event*.dot 2>/dev/null | \
  xargs -P "$JOBS" -I {} bash -c 'timeout 1500 "$ENGINE" -Tsvg "$1" -o "${1%.dot}.svg" 2>/dev/null' _ {}

echo "Gallery written to $OUT"
for d in "$OUT"/*/; do
  echo "  $(basename "$d"): $(ls "$d"/*.dot 2>/dev/null | wc -l) dot, $(ls "$d"/*.svg 2>/dev/null | wc -l) svg"
done
