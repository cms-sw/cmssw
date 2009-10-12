#! /bin/bash

# ConfDB configurations to use
MASTER="/dev/CMSSW_3_2_7/HLT"               # no explicit version, take te most recent 
TARGET="/dev/CMSSW_3_2_7/\$TABLE"           # no explicit version, take te most recent 
TABLES="8E29 1E31 GRun HIon"                # $TABLE in the above variable will be expanded to these TABLES

# getHLT.py
PACKAGE="HLTrigger/Configuration"
if [ -f "./getHLT.py" ]; then
  GETHLT="./getHLT.py"
elif [ -f "$CMSSW_BASE/src/$PACKAGE/test/getHLT.py" ]; then
  GETHLT="$CMSSW_BASE/src/$PACKAGE/test/getHLT.py"
elif [ -f "$CMSSW_RELEASE_BASE/src/$PACKAGE/test/getHLT.py" ]; then
  GETHLT="$CMSSW_RELEASE_BASE/src/$PACKAGE/test/getHLT.py"
else
  echo "cannot find getHLT.py, aborting"
  exit 1
fi

function getConfigForCVS() {
  # for things in CMSSW CVS
  local CONFIG="$1"
  local NAME="$2"
  $GETHLT $CONFIG $NAME GEN-HLT
}

function getOutputCommands() {
  # split the arguments using commas or spaces, and appends "::outputCommands" to each one
  echo $@ | sed -e's/\>/::outputCommands/g' -e's/[, ]\+/,/g'
}

function getContentForCVS() {
  # FIXME - the pipe through sed removes the definition of streams and primary datasets from the dump - this should be done directly by edmConfigFromDB
  local CONFIG="$1"
  local OUTPUT_ALCA="hltOutputALCAPHISYM hltOutputALCAPHISYMHCAL hltOutputALCAP0 hltOutputRPCMON"
  local OUTPUT_MON="hltOutputDQM hltOutputHLTDQM hltOutputHLTMON hltOutput8E29 hltOutput1E31 hltOutputHIon"
  edmConfigFromDB --configName $CONFIG --noedsources --nopaths --noes --nopsets --noservices --cff --blocks $(getOutputCommands "hltOutputA") --format python | sed -e'/^streams/,/^)/d' -e'/^datasets/,/^)/d' > hltOutputA_cff.py
  edmConfigFromDB --configName $CONFIG --noedsources --nopaths --noes --nopsets --noservices --cff --blocks $(getOutputCommands $OUTPUT_MON)  --format python | sed -e'/^streams/,/^)/d' -e'/^datasets/,/^)/d' > hltOutputMON_cff.py
  edmConfigFromDB --configName $CONFIG --noedsources --nopaths --noes --nopsets --noservices --cff --blocks $(getOutputCommands $OUTPUT_ALCA) --format python | sed -e'/^streams/,/^)/d' -e'/^datasets/,/^)/d' > hltOutputALCA_cff.py
}

function getConfigForOnline() {
  # for things NOT in CMSSW CVS:
  local CONFIG="$1"
  local NAME="$2"
  $GETHLT $CONFIG $NAME
}

# make sure we're using *this* working area
eval `scramv1 runtime -sh`
hash -r

if [ "$1" == "CVS" ]; then
  # for things in CMSSW CVS
  rm -f HLT*_cff.py
  getConfigForCVS  $MASTER FULL
  getContentForCVS $MASTER
  for TABLE in $TABLES; do
    getConfigForCVS $(eval echo $TARGET) $TABLE
  done
  ls -l HLT_*_cff.py hltOutput*_cff.py
  mv -f HLT_*_cff.py hltOutput*_cff.py ../python
else
  # for things NOT in CMSSW CVS:
  rm -f OnLine_HLT_*.py
  for TABLE in $TABLES; do
    getConfigForOnline $(eval echo $TARGET) $TABLE
  done
  ls -l OnLine_HLT_*.py
fi
