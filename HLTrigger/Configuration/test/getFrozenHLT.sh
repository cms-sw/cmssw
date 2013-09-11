#! /bin/bash

# ConfDB configurations to use
TABLES="8E33v2 8E33v2"
HLT_8E33v2="/online/collisions/2012/8e33/v2.2/HLT/V7"

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
  hltGetConfiguration --cff --offline --mc    $CONFIG --type "GRun" > HLT_${NAME}_cff.py
  hltGetConfiguration --fastsim               $CONFIG --type "GRun" > HLT_${NAME}_Famos_cff.py
}

function getConfigForOnline() {
  local CONFIG="$1"
  local NAME="$2"
  log "  dumping full HLT for $NAME from $CONFIG"
  # override the conditions with a menu-dependent "virtual" global tag, which takes care of overriding the L1 menu
  hltGetConfiguration --full --offline --data $CONFIG --type "GRun" --unprescale --process "HLT${NAME}" --globaltag "auto:hltonline_${NAME}" --input "file:RelVal_Raw_${NAME}_DATA.root"    > OnData_HLT_${NAME}.py
  hltGetConfiguration --full --offline --mc   $CONFIG --type "GRun" --unprescale --process "HLT${NAME}" --globaltag "auto:startup_${NAME}"   --input "file:RelVal_Raw_${NAME}_STARTUP.root" > OnLine_HLT_${NAME}.py
}

# make sure we're using *this* working area
eval `scramv1 runtime -sh`
hash -r

# cff python dumps, in CVS under HLTrigger/Configuration/pyhon
log "Extracting cff python dumps"
FILES=$(eval echo HLT_{$TABLES_}_cff.py HLT_{$TABLES_}_Famos_cff.py)
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
FILES=$(eval echo On{Data,Line}_HLT_{$TABLES_}.py)
rm -f $FILES
for TABLE in $TABLES; do
  CONFIG=$(eval echo \$$(echo HLT_$TABLE))
  getConfigForOnline $CONFIG $TABLE
done
log "Done"
log "$(ls -l $FILES)"
log
