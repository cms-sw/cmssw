#! /bin/bash

# ConfDB configurations to use
TABLES="Fake 50ns_5e33_v1 25ns14e33_v1 50ns_5e33_v3 25ns14e33_v3"
#TABLES="Fake 50ns_5e33_v1 25ns14e33_v1"
# Do not update the frozen menus in 74X
# TABLES="Fake Fake"
HLT_Fake="/dev/CMSSW_7_4_0/Fake"
HLT_50ns_5e33_v3="/frozen/2015/50ns_5e33/v3.0/HLT"
HLT_25ns14e33_v3="/frozen/2015/25ns14e33/v3.3/HLT"
HLT_50ns_5e33_v1="/frozen/2015/50ns_5e33/v1.2/HLT"
HLT_25ns14e33_v1="/frozen/2015/25ns14e33/v1.2/HLT"

# print extra messages ?
VERBOSE=false

# this is used for brace expansion
TABLES_=$(echo $TABLES | sed -e's/ \+/,/g')

[ "$1" == "-v" ] && { VERBOSE=true;  shift; }
[ "$1" == "-q" ] && { VERBOSE=false; shift; }

function log() {
  $VERBOSE && echo -e "$@"
}

function getConfigForCVS() {
  local CONFIG="$1"
  local NAME="$2"
  log "  dumping HLT cffs for $NAME from $CONFIG"
  # do not use any conditions or L1 override
  hltGetConfiguration --cff --offline --data  $CONFIG --type $NAME  > HLT_${NAME}_cff.py
}

function getConfigForOnline() {
  local CONFIG="$1"
  local NAME="$2"
  log "  dumping full HLT for $NAME from $CONFIG"
  # override the conditions with a menu-dependent "virtual" global tag, which takes care of overriding the L1 menu

  if [ "$NAME" == "Fake" ]; then
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process "HLT${NAME}" --globaltag "auto:run1_hlt_${NAME}" --input "file:RelVal_Raw_${NAME}_DATA.root" > OnLine_HLT_${NAME}.py
  else
    hltGetConfiguration --full --offline --data $CONFIG --type $NAME --unprescale --process "HLT${NAME}" --globaltag "auto:run2_hlt_${NAME}" --input "file:RelVal_Raw_${NAME}_DATA.root" > OnLine_HLT_${NAME}.py
  fi
}

# make sure we're using *this* working area
eval `scramv1 runtime -sh`
hash -r

# cff python dumps, in CVS under HLTrigger/Configuration/pyhon
log "Extracting cff python dumps"
FILES=$(eval echo HLT_{$TABLES_}_cff.py)
rm -f $FILES
for TABLE in $TABLES; do
  CONFIG=$(eval echo \$$(echo HLT_$TABLE))
  getConfigForCVS    $CONFIG $TABLE
done
log "Done"
log "$(ls -l $FILES)"
mv -f $FILES ../python/
log

# full config dumps, in CVS under HLTrigger/Configuration/test
log "Extracting full configuration dumps"
FILES=$(eval echo OnLine_HLT_{$TABLES_}.py)
rm -f $FILES
for TABLE in $TABLES; do
  CONFIG=$(eval echo \$$(echo HLT_$TABLE))
  getConfigForOnline $CONFIG $TABLE
done
log "Done"
log "$(ls -l $FILES)"
log
