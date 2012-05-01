#! /bin/bash

# ConfDB configurations to use
MASTER="/dev/CMSSW_4_2_0/HLT"        # no explicit version, take te most recent
TARGET="/dev/CMSSW_4_2_0/\$TABLE"    # no explicit version, take te most recent
TABLES="GRun HIon"                   # $TABLE in the above variable will be expanded to these TABLES

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

  # do not use any L1 override
  hltGetConfiguration --cff --offline --mc   $CONFIG --type $NAME > HLT_${NAME}_cff.py
  hltGetConfiguration --cff --offline --data $CONFIG --type $NAME > HLT_${NAME}_data_cff.py
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
# local L1T="tag[,connect]" - record is hardwired as L1GtTriggerMenuRcd
  local L1TPP="L1GtTriggerMenu_L1Menu_Collisions2011_v6_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2011_v6/sqlFile/L1Menu_Collisions2011_v6_mc.db"
# local L1TPP="L1GtTriggerMenu_L1Menu_Collisions2011_v6_mc"
  local L1THI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2010_v2_mc"


  log "    dumping full HLT for $NAME"
  # override L1 menus
  if [ "$NAME" == "GRun" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1TPP --globaltag auto:hltonline     > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1TPP --globaltag auto:startup       > OnLine_HLT_$NAME.py 
  elif [ "$NAME" == "HIon" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1THI --globaltag auto:hltonline     > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1THI --globaltag auto:starthi       > OnLine_HLT_$NAME.py
  else
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME             --globaltag auto:hltonline     > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME                                            > OnLine_HLT_$NAME.py
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
