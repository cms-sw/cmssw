#! /bin/bash

function getConfigForCVS() {
  # for things in CMSSW CVS
  local HLTtable="$1"
  local HLTid="$2"
  ./getHLT.py $HLTtable $HLTid GEN-HLT
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
  ./getHLT.py $HLTtable $HLTid
}



eval `scramv1 runtime -sh`
hash -r

HLTtable8E29="/dev/CMSSW_2_2_6_HLT/8E29/V5"
HLTtable1E31="/dev/CMSSW_2_2_6_HLT/1E31/V5"
HLTcontent="/dev/CMSSW_2_2_6_HLT/merged/V4"

if [ "$1" == "CVS" ]; then
  # for things in CMSSW CVS
  getConfigForCVS "${HLTtable8E29}" "8E29"
  getConfigForCVS "${HLTtable1E31}" "1E31"
  getContentForCVS "${HLTcontent}"

  ls -lt HLT*_cff.py
  mv -f HLT*_cff.py ../python
else
  # for things NOT in CMSSW CVS:
  getConfigForOnline "${HLTtable_8E29}" "8E29"
  getConfigForOnline "${HLTtable_1E31}" "1E31"
fi
