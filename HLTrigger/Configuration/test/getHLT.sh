#! /bin/bash

# ConfDB configurations to use
MASTER="/dev/CMSSW_3_4_0/pre1/HLT"          # no explicit version, take te most recent 
TARGET="/dev/CMSSW_3_4_0/pre1/\$TABLE"      # no explicit version, take te most recent 
TABLES="8E29 1E31 GRun HIon"           # $TABLE in the above variable will be expanded to these TABLES

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

if [ -f "./getEventContent.py" ]; then
  GETCONTENT="./getEventContent.py"
elif [ -f "$CMSSW_BASE/src/$PACKAGE/test/getEventContent.py" ]; then
  GETCONTENT="$CMSSW_BASE/src/$PACKAGE/test/getEventContent.py"
elif [ -f "$CMSSW_RELEASE_BASE/src/$PACKAGE/test/getEventContent.py" ]; then
  GETCONTENT="$CMSSW_RELEASE_BASE/src/$PACKAGE/test/getEventContent.py"
else
  echo "cannot find getEventContent.py, aborting"
  exit 1
fi

function getConfigForCVS() {
  # for things in CMSSW CVS
  local CONFIG="$1"
  local NAME="$2"
  $GETHLT $CONFIG $NAME GEN-HLT
}

function getContentForCVS() {
  local CONFIG="$1"

  $GETCONTENT $CONFIG
  rm -f hltOutputA_cff.pyc hltOutputMON_cff.pyc hltOutputALCA_cff.pyc
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
  ls -l HLT_*_cff.py hltOutput*_cff.py HLTrigger_EventContent_cff.py
  mv -f HLT_*_cff.py hltOutput*_cff.py HLTrigger_EventContent_cff.py ../python
else
  # for things NOT in CMSSW CVS:
  rm -f OnLine_HLT_*.py
  for TABLE in $TABLES; do
    getConfigForOnline $(eval echo $TARGET) $TABLE
  done
  ls -l OnLine_HLT_*.py
fi
