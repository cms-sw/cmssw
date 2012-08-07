#! /bin/bash

# ConfDB configurations to use
TABLES="5E33v4 7E33v2 7E33v3 7E33v4"
HLT_5E33v4="/online/collisions/2012/5e33/v4.4/HLT/V8"
HLT_7E33v2="/online/collisions/2012/7e33/v2.2/HLT/V6"
HLT_7E33v3="/online/collisions/2012/7e33/v3.0/HLT/V21"
HLT_7E33v4="/online/collisions/2012/7e33/v4.1/HLT/V2"

# print extra messages ?
VERBOSE=false

# this is used for brace expansion
TABLES_=$(echo $TABLES | sed -e's/ \+/,/g')

[ "$1" == "-v" ] && { VERBOSE=true; shift; }

function log() {
  $VERBOSE && echo -e "$@"
}

function getConfigForCVS() {
  local CONFIG="$1"
  local NAME="$2"
  log "    dumping HLT cffs for $NAME from $CONFIG"
  # do not use any conditions or L1 override
  log "    hltGetConfiguration --cff --offline --mc    $CONFIG --type GRun > HLT_${NAME}_cff.py"
  hltGetConfiguration --cff --offline --mc    $CONFIG --type GRun > HLT_${NAME}_cff.py
  log "    hltGetConfiguration --fastsim               $CONFIG --type GRun > HLT_${NAME}_Famos_cff.py"
  hltGetConfiguration --fastsim               $CONFIG --type GRun > HLT_${NAME}_Famos_cff.py
}

function getConfigForOnline() {
  local CONFIG="$1"
  local NAME="$2"
  log "    dumping full HLT for $NAME from $CONFIG"
  # override the conditions with a menu-dependent "virtual" global tag, which takes care of overriding the L1 menu
  log "    hltGetConfiguration --full --offline --data $CONFIG --type GRun --unprescale --process HLT$NAME --globaltag auto:hltonline_$NAME > OnData_HLT_${NAME}.py"
  hltGetConfiguration --full --offline --data $CONFIG --type GRun --unprescale --process HLT$NAME --globaltag auto:hltonline_$NAME > OnData_HLT_${NAME}.py
  log "    hltGetConfiguration --full --offline --mc   $CONFIG --type GRun --unprescale --process HLT$NAME --globaltag auto:startup_$NAME   > OnLine_HLT_${NAME}.py"
  hltGetConfiguration --full --offline --mc   $CONFIG --type GRun --unprescale --process HLT$NAME --globaltag auto:startup_$NAME   > OnLine_HLT_${NAME}.py
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
FILES=$(eval echo OnData_HLT_{$TABLES_}.py OnLine_HLT_{$TABLES_}.py)
rm -f $FILES
for TABLE in $TABLES; do
  CONFIG=$(eval echo \$$(echo HLT_$TABLE))
  getConfigForOnline $CONFIG $TABLE
done
log "Done"
log "$(ls -l $FILES)"
log
