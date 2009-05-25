#! /bin/bash

# ConfDB configurations to use
HLTtable2E30="/dev/CMSSW_2_2_9/HLT/V2"

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

function getConfigForCVS() {
  # for things in CMSSW CVS
  local HLTtable="$1"
  local HLTid="$2"
  $GETHLT $HLTtable $HLTid GEN-HLT
}

function getContentForCVS() {
  local HLTcontent="$1"
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDefaultOutput::outputCommands         --format python > HLTDefaultOutput_cff.py
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDefaultOutputWithFEDs::outputCommands --format python > HLTDefaultOutputWithFEDs_cff.py
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDebugOutput::outputCommands           --format python > HLTDebugOutput_cff.py
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDebugWithAlCaOutput::outputCommands   --format python > HLTDebugWithAlCaOutput_cff.py
}

function getConfigForOnline() {
  # for things NOT in CMSSW CVS:
  local HLTtable="$1"
  local HLTid="$2"
  $GETHLT $HLTtable $HLTid
}

# make sure we're using *this* working area
eval `scramv1 runtime -sh`
hash -r

if [ "$1" == "CVS" ]; then
  # for things in CMSSW CVS
  rm -f HLT*_cff.py
  getConfigForCVS "${HLTtable2E30}" "2E30"
  getContentForCVS "${HLTtable2E30}"
  ls -lt HLT*_cff.py
  mv -f HLT*_cff.py ../python
else
  # for things NOT in CMSSW CVS:
  rm -f  OnLine_HLT_*.py
  getConfigForOnline "${HLTtable2E30}" "2E30"
  ls -lt OnLine_HLT_*.py
fi
