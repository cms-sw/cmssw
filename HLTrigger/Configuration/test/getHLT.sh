#! /bin/bash

# ConfDB configurations to use
MASTER="/dev/CMSSW_3_6_0/patch1/HLT"       # no explicit version, take te most recent 
TARGET="/dev/CMSSW_3_6_0/patch1/\$TABLE"   # no explicit version, take te most recent 
TABLES="8E29 1E31 GRun HIon"               # $TABLE in the above variable will be expanded to these TABLES

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

if [ -f "./getDatasets.py" ]; then
  GETDATASETS="./getDatasets.py"
elif [ -f "$CMSSW_BASE/src/$PACKAGE/test/getDatasets.py" ]; then
  GETDATASETS="$CMSSW_BASE/src/$PACKAGE/test/getDatasets.py"
elif [ -f "$CMSSW_RELEASE_BASE/src/$PACKAGE/test/getDatasets.py" ]; then
  GETDATASETS="$CMSSW_RELEASE_BASE/src/$PACKAGE/test/getDatasets.py"
else
  echo "cannot find getDatasets.py, aborting"
  exit 1
fi

function getConfigForCVS() {
  # for things in CMSSW CVS
  local CONFIG="$1"
  local NAME="$2"
  if [ "${NAME}" == "8E29" ] || [ "${NAME}" == "GRun" ]; then
   $GETHLT --cff --mc --l1 L1GtTriggerMenu_L1Menu_Commissioning2010_v1_mc $CONFIG $NAME
  else
   $GETHLT --cff --mc $CONFIG $NAME
  fi
}

function getContentForCVS() {
  local CONFIG="$1"

  $GETCONTENT $CONFIG
  rm -f hltOutput*_cff.py*
}

function getDatasetsForCVS() {
  local CONFIG="$1"
  local TARGET="$2"

  $GETDATASETS $CONFIG $TARGET
}

function getConfigForOnline() {
  # for things NOT in CMSSW CVS:
  local CONFIG="$1"
  local NAME="$2"
  if [ "${NAME}" == "8E29" ] || [ "${NAME}" == "GRun" ]; then
   $GETHLT --full --offline --data --l1 L1GtTriggerMenu_L1Menu_Commissioning2010_v1_mc $CONFIG $NAME
   $GETHLT --full --offline --mc   --l1 L1GtTriggerMenu_L1Menu_Commissioning2010_v1_mc $CONFIG $NAME
  else
   $GETHLT --full --offline --data $CONFIG $NAME
   $GETHLT --full --offline --mc   $CONFIG $NAME
  fi
}

# make sure we're using *this* working area
eval `scramv1 runtime -sh`
hash -r

# for things in CMSSW CVS
echo "Extracting CVS python dumps"
rm -f HLT*_cff.py
getConfigForCVS  $MASTER FULL
getContentForCVS $MASTER
for TABLE in $TABLES; do
  getConfigForCVS $(eval echo $TARGET) $TABLE
done
for TABLE in "GRun"; do
  getDatasetsForCVS $(eval echo $TARGET) HLTrigger_Datasets_cff.py
done
ls -l HLT_*_cff.py HLTrigger_EventContent_cff.py HLTrigger_Datasets_cff.py
mv -f HLT_*_cff.py HLTrigger_EventContent_cff.py HLTrigger_Datasets_cff.py ../python
echo

# for things now also in CMSSW CVS:
echo "Extracting full configurations"
rm -f OnData_HLT_*.py
rm -f OnLine_HLT_*.py
for TABLE in $TABLES; do
  getConfigForOnline $(eval echo $TARGET) $TABLE
done
ls -l OnData_HLT_*.py
ls -l OnLine_HLT_*.py
echo
