#!/bin/bash
SEQUENCE=""
ERA="Run2_2018"
SCENARIO="pp"
STEP="DQM"

while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
    echo "Usage: $0 [ --era ERA ] [ --scenario SCENARIO ] [ --step STEP ] { SEQUENCNAME | @AUTODQMSEQUENCE }"
    echo "STEP can be 'DQM', 'VALIDATION' or 'HARVESTING' (others may or may not work) (default '$STEP')"
    echo "SCENARIO can be pp, cosmics, or HeavyIons (default '$SCENARIO')"
    echo "ERA can be any era (default '$ERA')"
    exit ;;
    --era)
    ERA="$2"; shift 2 ;;
    --scenario)
    SCENARIO="$2"; shift 2 ;;
    --step)
    STEP="$2"; shift 2 ;;
    --*)
    echo "Option '$1' not recognized." 1>&2
    exit 1 ;;
    *)
    SEQUENCE="$1"; shift 1 ;;
  esac
done

if [[ -z  $SEQUENCE ]]; then
  SEP=""
else
  SEP=":"
fi

if [[ $STEP == "DQM" ]]; then
  TYPE='--data'
fi

if [[ $STEP != "HARVESTING" ]]; then
  OTHERSTEP='RAW2DIGI:siPixelDigis,'
fi

# This file will actually be opened, though the content does not matter. Only to make CMSSW start up at all.
INFILE="/store/data/Run2018A/EGamma/RAW/v1/000/315/489/00000/004D960A-EA4C-E811-A908-FA163ED1F481.root"
# Modules that will be loaded but do not come from the DQM Sequence.
BLACKLIST='^(TriggerResults|.*_step|DQMoutput|siPixelDigis)$'

cmsdriverargs=(
  step3
  --conditions auto:run2_data                                         # conditions is mandatory, but should not affect the result.
  -s "$OTHERSTEP$STEP$SEP$SEQUENCE"                                   # running only DQM seems to be not possible, so also load a single module for RAW2DIGI
  --process DUMMY $TYPE --era "$ERA"                                  # random switches, era is important as it trigger e.g. switching phase0/pahse1/phase2
  --eventcontent DQM --scenario $SCENARIO --datatier DQMIO            # more random switches, sceanario should affect which DQMOffline_*_cff.py is loaded
  --customise_commands 'process.Tracer = cms.Service("Tracer")'       # the tracer will tell us which modules actually run
  --runUnscheduled                                                    # convert everything to tasks. Used in production, which means sequence ordering does not matter!
  --filein "$INFILE" -n 0                                             # load an input file, but do not process any events -- it would fail anyways.
)

echo Running cmsDriver.py "${cmsdriverargs[@]}" 1>&2

# Finally run cmssw and select the modules names out of the tracer output.
# Remove some modules that are there for technical reasons but did not come from the DQM sequences.
# Sorted output for easier diff'ing, ordering should not matter anyways.
cmsDriver.py "${cmsdriverargs[@]}"  2>&1 \
  | grep "++++ starting: constructing module with label" | grep -oE "'[^']*'" | tr -d "'" \
  | grep -vE "$BLACKLIST" \
  | sort
