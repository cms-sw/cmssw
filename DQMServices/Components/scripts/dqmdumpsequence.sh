#!/bin/bash
SEQUENCE=""
ERA="Run2_2018"
SCENARIO="pp"
STEP="DQM"
ANALYZE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
    echo "Usage: $0 [--analyze-modules] [ --era ERA ] [ --scenario SCENARIO ] [ --step STEP ] { SEQUENCNAME | @AUTODQMSEQUENCE }"
    echo "STEP can be 'DQM', 'VALIDATION' or 'HARVESTING' (others may or may not work) (default '$STEP')"
    echo "SCENARIO can be pp, cosmics, or HeavyIons (default '$SCENARIO')"
    echo "ERA can be any era (default '$ERA')"
    echo "If --analyze-modules is passed, run the modules through the equivalent of edmCheckMultithreading."
    echo
    echo "Example:"
    cat <<'EOF'
    # Analyze modules on all DQM and VALIDATION sequences.
    for seq in $(cat Validation/Configuration/python/autoValidation.py | grep -oE "'\w+' ?:" | tr -d "':" | sort | uniq); do (mkdir -p validation/$seq; cd validation/$seq; dqmdumpsequence.sh --analyze-modules --step VALIDATION @$seq > list ) & done
    for seq in $(cat DQMOffline/Configuration/python/autoDQM.py | grep -oE "'\w+' ?:" | tr -d "':" | sort | uniq); do (mkdir -p dqm/$seq; cd dqm/$seq; dqmdumpsequence.sh --analyze-modules --step DQM @$seq > list ) & done
EOF
    exit ;;
    --era)
    ERA="$2"; shift 2 ;;
    --scenario)
    SCENARIO="$2"; shift 2 ;;
    --step)
    STEP="$2"; shift 2 ;;
    --analyze-modules)
    ANALYZE="yes"; shift ;;
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
cmsDriver.py --python_filename cmssw_cfg.py "${cmsdriverargs[@]}" --no_exec 1>&2
cmsRun cmssw_cfg.py 2>&1 \
  | grep "++++ starting: constructing module with label" | grep -oE "'[^']*'" | tr -d "'" \
  | grep -vE "$BLACKLIST" \
  | sort | tee modlist

if [[ -z $ANALYZE ]]; then
  exit 0
fi

# with the --analyze-modules option, we now run a tuned edmCheckMultithreading on the list of modules.
# The hard part is that we have module labels, but we need module class names. So, we still need to 
# run edmConfigDump (as edmCheckMultithreading usually does), but filter down the output using our 
# list of modules.

# use colors only if the output is sent to a terminal
if [ -t 1 ] ; then
  SETCOLOR_SUCCESS='\e[0;32m'
  SETCOLOR_WARNING='\e[1;33m'
  SETCOLOR_FAILURE='\e[0;31m'
  SETCOLOR_NORMAL_='\e[0m'
else
  SETCOLOR_SUCCESS=''
  SETCOLOR_WARNING=''
  SETCOLOR_FAILURE=''
  SETCOLOR_NORMAL_=''
fi

edmConfigDump cmssw_cfg.py | fgrep -f modlist | grep 'cms\.EDAnalyzer\|cms\.EDFilter\|cms\.EDProducer\|cms\.OutputModule' | cut -d'"' -f2 | sort -u | while read MODULE
do
  edmPluginHelp -p $MODULE | {
    read N CLASS TYPE P
    read
    if [ ! "$CLASS" ]; then
      FILL="--"
      TYPE="--"
      MT="--"
      ED="--"
    else
      if grep -q "not implemented the function"; then
        FILL="no"
        SETCOLOR_FILL="$SETCOLOR_WARNING"
      else
        FILL="yes"
        SETCOLOR_FILL="$SETCOLOR_SUCCESS"
      fi
      TYPE=`echo $TYPE | sed -e's/[()]//g'`
      MT=`echo $TYPE | cut -s -d: -f1`
      ED=`echo $TYPE | cut -s -d: -f3`
      if ! [ "$MT" ]; then
        MT="legacy"
        ED="$TYPE"
        SETCOLOR_MT="$SETCOLOR_FAILURE"
      elif [ "$MT" == "one" ]; then
        SETCOLOR_MT="$SETCOLOR_WARNING"
      else
        SETCOLOR_MT="$SETCOLOR_SUCCESS"
      fi
    fi
    printf "%-64s%-16s%b%-16s%b%s$SETCOLOR_NORMAL_\n" "$MODULE" "$ED" "$SETCOLOR_MT" "$MT" "$SETCOLOR_FILL" "$FILL"
  }
done
