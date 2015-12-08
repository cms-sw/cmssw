#! /bin/bash

# ConfDB configurations to use
MASTER="/dev/CMSSW_7_5_0/HLT"              # no explicit version, take te most recent
TARGET="/dev/CMSSW_7_5_0/\$TABLE"          # no explicit version, take te most recent
#TABLES="GRun HIon PIon PRef 50nsGRun LowPU 25nsLowPU"     # $TABLE in the above variable will be expanded to these TABLES
TABLES="GRun HIon PIon PRef"     # $TABLE in the above variable will be expanded to these TABLES

# print extra messages ?
VERBOSE=false

# this is used for brace expansion
TABLES_=$(echo $TABLES | sed -e's/ \+/,/g')

[ "$1" == "-v" ] && { VERBOSE=true;  shift; }
[ "$1" == "-q" ] && { VERBOSE=false; shift; }

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
  log "  dumping HLT cffs for $NAME from $CONFIG"

  # do not use any conditions or L1 override
  hltGetConfiguration --cff --offline --data $CONFIG --type $NAME > HLT_${NAME}_cff.py
}

function getContentForCVS() {
  local CONFIG="$1"

  log "  dumping EventContet"
  $GETCONTENT $CONFIG
  rm -f hltOutput*_cff.py*
}

function getDatasetsForCVS() {
  local CONFIG="$1"
  local TARGET="$2"

  log "  dumping Primary Dataset"
  $GETDATASETS $CONFIG $TARGET
}

function getConfigForOnline() {
  local CONFIG="$1"
  local NAME="$2"
# local L1T="tag[,connect]" - record is hardwired as L1GtTriggerMenuRcd

# local L1TPP="L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2012_v3/sqlFile/L1Menu_Collisions2012_v3_mc.db"
# local L1TPP="L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc"
# local L1TPP="L1GtTriggerMenu_L1Menu_Collisions2015_25ns_v1_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2015_25ns_v1/sqlFile/L1Menu_Collisions2015_25ns_v1_mc.db"
# local L1THI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2011_v0/sqlFile/L1Menu_CollisionsHeavyIons2011_v0_mc.db"
# local L1THI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc"
# local L1THI="L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc"
# local L1THI="L1GtTriggerMenu_L1Menu_Collisions2015_25ns_v1_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2015_25ns_v1/sqlFile/L1Menu_Collisions2015_25ns_v1_mc.db"
# local L1TPI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2013_v0/sqlFile/L1Menu_CollisionsHeavyIons2013_v0_mc.db"
# local L1TPI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc"
# local L1TPI="L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc"
# local L1TPI="L1GtTriggerMenu_L1Menu_Collisions2015_25ns_v1_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2015_25ns_v1/sqlFile/L1Menu_Collisions2015_25ns_v1_mc.db"

  local L1TPP1=""
  local L1TPP2=""

  log "  dumping full HLT for $NAME from $CONFIG"
  # override L1 menus
  if [ "$NAME" == "Fake" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --globaltag "auto:run1_hlt_${NAME}" --input "file:RelVal_Raw_${NAME}_DATA.root" > OnLine_HLT_$NAME.py
  else
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --globaltag "auto:run2_hlt_${NAME}" --input "file:RelVal_Raw_${NAME}_DATA.root" > OnLine_HLT_$NAME.py
  fi

}

# make sure we're using *this* working area
eval `scramv1 runtime -sh`
hash -r

# cff python dumps, in CVS under HLTrigger/Configuration/pyhon
log "Extracting cff python dumps"
FILES=$(eval echo HLT_FULL_cff.py HLT_{$TABLES_}_cff.py HLTrigger_Datasets_{$TABLES_}_cff.py HLTrigger_EventContent_cff.py )
rm -f $FILES
getConfigForCVS  $MASTER FULL
getContentForCVS $MASTER
for TABLE in $TABLES; do
  getConfigForCVS $(eval echo $TARGET) $TABLE
  getDatasetsForCVS $(eval echo $TARGET) HLTrigger_Datasets_${TABLE}_cff.py
done
log "Done"
log "$(ls -l $FILES)"
mv -f $FILES ../python/
log

# full config dumps, in CVS under HLTrigger/Configuration/test
log "Extracting full configuration dumps"
FILES=$(eval echo OnLine_HLT_FULL.py OnLine_HLT_{$TABLES_}.py)
rm -f $FILES
getConfigForOnline $MASTER FULL
for TABLE in $TABLES; do
  getConfigForOnline $(eval echo $TARGET) $TABLE
done
log "Done"
log "$(ls -l $FILES)"
log
