#! /bin/bash

# ConfDB configurations to use
MASTER="/dev/CMSSW_5_2_6/HLT"        # no explicit version, take te most recent
TARGET="/dev/CMSSW_5_2_6/\$TABLE"    # no explicit version, take te most recent
#TABLES="GRun HIon PIon"              # $TABLE in the above variable will be expanded to these TABLES
TABLES="PIon"              # $TABLE in the above variable will be expanded to these TABLES

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
  if [ "$NAME" == "GRun" ]; then
    local L1XmlPP="L1Menu_Collisions2012_v3_L1T_Scales_20101224_Imp0_0x102b.xml"
    hltGetConfiguration --cff --offline --mc   $CONFIG --type $NAME --l1Xml $L1XmlPP > HLT_${NAME}_cff.py
    hltGetConfiguration --cff --offline --data $CONFIG --type $NAME --l1Xml $L1XmlPP > HLT_${NAME}_data_cff.py
    hltGetConfiguration --fastsim              $CONFIG --type $NAME --l1Xml $L1XmlPP > HLT_${NAME}_Famos_cff.py
  elif [ "$NAME" == "HIon" ]; then
    hltGetConfiguration --cff --offline --mc   $CONFIG --type $NAME                > HLT_${NAME}_cff.py
    hltGetConfiguration --cff --offline --data $CONFIG --type $NAME                > HLT_${NAME}_data_cff.py
    hltGetConfiguration --fastsim              $CONFIG --type $NAME                > HLT_${NAME}_Famos_cff.py
  elif [ "$NAME" == "PIon" ]; then
    local L1XmlPI="L1Menu_CollisionsHeavyIons2013_v0_L1T_Scales_20101224_Imp0_0x102c.xml"
    hltGetConfiguration --cff --offline --mc   $CONFIG --type $NAME  --l1Xml $L1XmlPI > HLT_${NAME}_cff.py
    hltGetConfiguration --cff --offline --data $CONFIG --type $NAME  --l1Xml $L1XmlPI > HLT_${NAME}_data_cff.py
    hltGetConfiguration --fastsim              $CONFIG --type $NAME  --l1Xml $L1XmlPI > HLT_${NAME}_Famos_cff.py
  else
    hltGetConfiguration --cff --offline --mc   $CONFIG --type $NAME                > HLT_${NAME}_cff.py
    hltGetConfiguration --cff --offline --data $CONFIG --type $NAME                > HLT_${NAME}_data_cff.py
    hltGetConfiguration --fastsim              $CONFIG --type $NAME                > HLT_${NAME}_Famos_cff.py
  fi
  diff -C0 HLT_${NAME}_cff.py HLT_${NAME}_data_cff.py
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
  local L1TPP="L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2012_v3/sqlFile/L1Menu_Collisions2012_v3_mc.db"
# local L1TPP="L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc"
# local L1THI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_CollisionsHeavyIons2011_v0/sqlFile/L1Menu_CollisionsHeavyIons2011_v0_mc.db"
  local L1THI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc"
  local L1TPI="L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2012_v3/sqlFile/L1Menu_CollisionsHeavyIons2013_v0_mc.db"
# local L1TPI="L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc"

  local L1XmlPP="L1Menu_Collisions2012_v3_L1T_Scales_20101224_Imp0_0x102b.xml"
  local L1XmlPI="L1Menu_CollisionsHeavyIons2013_v0_L1T_Scales_20101224_Imp0_0x102c.xml"

  log "    dumping full HLT for $NAME"
  # override L1 menus
  if [ "$NAME" == "GRun" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --l1Xml $L1XmlPP --l1-emulator --globaltag auto:hltonline_GRun     > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME --l1Xml $L1XmlPP --l1-emulator --globaltag auto:startup_GRun       > OnLine_HLT_$NAME.py 
  elif [ "$NAME" == "HIon" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1THI --globaltag auto:hltonline_HIon     > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME --l1 $L1THI --globaltag auto:starthi_HIon       > OnLine_HLT_$NAME.py
  elif [ "$NAME" == "PIon" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process HLT$NAME --l1Xml $L1XmlPI --globaltag auto:hltonline_PIon     > OnData_HLT_$NAME.py
    hltGetConfiguration --full --offline --mc   $CONFIG --type $NAME --unprescale --process HLT$NAME --l1Xml $L1XmlPI --globaltag auto:startup_PIon       > OnLine_HLT_$NAME.py
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
  getDatasetsForCVS $(eval echo $TARGET) HLTrigger_Datasets_${TABLE}_cff.py
done
log "Done"
ls -l HLT_*_cff.py HLTrigger_EventContent_cff.py HLTrigger_Datasets_*_cff.py
mv -f HLT_*_cff.py HLTrigger_EventContent_cff.py HLTrigger_Datasets_*_cff.py ../python/
echo

# for things now also in CMSSW CVS:
echo "Extracting full configurations"
rm -f OnData_HLT_GRun_*.py
rm -f OnData_HLT_HIon_*.py
rm -f OnData_HLT_PIon_*.py
rm -f OnLine_HLT_GRun_*.py
rm -f OnLine_HLT_HIon_*.py
rm -f OnLine_HLT_PIon_*.py
for TABLE in $TABLES; do
  getConfigForOnline $(eval echo $TARGET) $TABLE
done
log "Done"
ls -l OnData_HLT_*.py
ls -l OnLine_HLT_*.py
echo
