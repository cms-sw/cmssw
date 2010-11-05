#! /bin/bash

# ConfDB configurations to use
MASTER="/dev/CMSSW_3_8_5/HLT"           # no explicit version, take te most recent 
TARGET="/dev/CMSSW_3_8_5/\$TABLE"       # no explicit version, take te most recent 
TABLES="GRun HIon"                      # $TABLE in the above variable will be expanded to these TABLES

# print extra messages ?
VERBOSE=false

[ "$1" == "-v" ] && { VERBOSE=true; shift; }

function log() {
  $VERBOSE && echo -e "$@"
}

function findHltScript() {
  local PACKAGE="HLTrigger/Configuration"
  local SCRIPT="$1"

  if [ -f "$SCRIPT" ]; then
    echo "./$SCRIPT"
  elif [ -f "$CMSSW_BASE/src/$PACKAGE/test/$SCRIPT" ]; then
    echo "$CMSSW_BASE/src/$PACKAGE/test/$SCRIPT"
  elif [ -f "$CMSSW_RELEASE_BASE/src/$PACKAGE/test/$SCRIPT" ]; then
    echo "$CMSSW_RELEASE_BASE/src/$PACKAGE/test/$SCRIPT"
  else
    echo "cannot find $SCRIPT, aborting" 
    exit 1
  fi
}

GETCONTENT=$(findHltScript getEventContent.py)
GETDATASETS=$(findHltScript getDatasets.py)

function getConfigForCVS() {
  local CONFIG="$1"
  local NAME="$2"
  log "    dumping HLT cff for $NAME"

  # override L1 menus
  if [ "$NAME" == "8E29" ] || [ "$NAME" == "GRun" ]; then
    hltGetConfiguration --cff --offline --mc $CONFIG --type $NAME                                  > HLT_${NAME}_cff.py
  elif [ "$NAME" == "1E31" ] || [ "$NAME" == "HIon" ]; then
    hltGetConfiguration --cff --offline --mc $CONFIG --type $NAME                                  > HLT_${NAME}_cff.py
  else
    hltGetConfiguration --cff --offline --mc $CONFIG --type $NAME                                  > HLT_${NAME}_cff.py
  fi

  # do not use any L1 override
  #hltGetConfiguration --cff --offline --mc $CONFIG --type $NAME > HLT_${NAME}_cff.py
}

function getContentForCVS() {
  local CONFIG="$1"

  log "    dumping EventContet"
  $GETCONTENT $CONFIG
  rm -f hltOutput*_cff.py*
}

function getDatasetsForCVS() {
  local CONFIG="$1"
  local TARGET="$2"

  log "    dumping Primary Dataset"
  $GETDATASETS $CONFIG $TARGET
}

function getConfigForOnline() {
  local CONFIG="$1"
  local NAME="$2"
  local L1T="L1Menu_Collisions2010_v0"

  log "    dumping full HLT for $NAME"
  # override L1 menus
  if [ "$NAME" == "8E29" ] || [ "$NAME" == "GRun" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1T                        --globaltag auto:hltonline > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1T                                                   > OnLine_HLT_$NAME.py 
  elif [ "$NAME" == "1E31" ] || [ "$NAME" == "HIon" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1T                        --globaltag auto:hltonline > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1T                                                   > OnLine_HLT_$NAME.py
  else
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME                                  --globaltag auto:hltonline > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME                                                             > OnLine_HLT_$NAME.py
  fi

  # do not use any L1 override
  #hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --globaltag auto:hltonline > OnData_HLT_$NAME.py
  #hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME                            > OnLine_HLT_$NAME.py
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
log "Done"
ls -l HLT_*_cff.py HLTrigger_EventContent_cff.py HLTrigger_Datasets_cff.py
mv -f HLT_*_cff.py HLTrigger_EventContent_cff.py HLTrigger_Datasets_cff.py ../python/
echo

# for things now also in CMSSW CVS:
echo "Extracting full configurations"
rm -f OnData_HLT_*.py
rm -f OnLine_HLT_*.py
for TABLE in $TABLES; do
  getConfigForOnline $(eval echo $TARGET) $TABLE
done
log "Done"
ls -l OnData_HLT_*.py
ls -l OnLine_HLT_*.py
echo
