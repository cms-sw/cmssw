#! /bin/bash

# ConfDB configurations to use
TABLES="Fake Fake1 Fake2 2018v32"
HLT_Fake="/dev/CMSSW_10_1_0/Fake"
HLT_Fake1="/dev/CMSSW_10_1_0/Fake1"
HLT_Fake2="/dev/CMSSW_10_1_0/Fake2"
HLT_2018v32="/frozen/2018/2e34/v3.2/HLT"

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
  hltGetConfiguration --cff --data $CONFIG --type $NAME  > HLT_${NAME}_cff.py
}

function getConfigForOnline() {
  local CONFIG="$1"
  local NAME="$2"
  log "  dumping full HLT for $NAME from $CONFIG"
  # override the conditions with a menu-dependent "virtual" global tag, which takes care of overriding the L1 menu

  if [ "$NAME" == "Fake" ]; then
    hltGetConfiguration --full --data $CONFIG --type $NAME --unprescale --process "HLT${NAME}" --globaltag "auto:run1_hlt_${NAME}" --input "file:RelVal_Raw_${NAME}_DATA.root" > OnLine_HLT_${NAME}.py
  else
    hltGetConfiguration --full --data $CONFIG --type $NAME --unprescale --process "HLT${NAME}" --globaltag "auto:run2_hlt_${NAME}" --input "file:RelVal_Raw_${NAME}_DATA.root" > OnLine_HLT_${NAME}.py
  fi
}

# make sure we're using *this* working area
eval `scramv1 runtime -sh`
hash -r

# cff python dumps, in CVS under HLTrigger/Configuration/pyhon
log "Extracting cff python dumps"
echo "Extracting cff python dumps"
FILES=$(eval echo HLT_{$TABLES_}_cff.py)
rm -f $FILES
for TABLE in $TABLES; do
  log "$TABLE"
  echo "$TABLE"
  CONFIG=$(eval echo \$$(echo HLT_$TABLE))
  getConfigForCVS    $CONFIG $TABLE
done
log "Done"
log "$(ls -l $FILES)"
mv -f $FILES ../python/
log

# full config dumps, in CVS under HLTrigger/Configuration/test
log "Extracting full configuration dumps"
echo "Extracting full configuration dumps"
FILES=$(eval echo OnLine_HLT_{$TABLES_}.py)
rm -f $FILES
for TABLE in $TABLES; do
  log "$TABLE"
  echo "$TABLE"
  CONFIG=$(eval echo \$$(echo HLT_$TABLE))
  getConfigForOnline $CONFIG $TABLE
done
log "Done"
log "$(ls -l $FILES)"
log
