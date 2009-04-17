#! /bin/bash

eval `scramv1 runtime -sh`
hash -r

if [ "$1" == "2E30" ]; then
  # get thje configuration for 2E30
  HLTid="2E30"
  HLTtable="/dev/CMSSW_2_2_6/HLT/V3"
  HLTcontent="/dev/CMSSW_2_2_6/HLT/V3"
  shift
else
  # get the default configuration (2E30)
  HLTid="2E30"
  HLTtable="/dev/CMSSW_2_2_6/HLT/V3"
  HLTcontent="/dev/CMSSW_2_2_6/HLT/V3"
fi

if [ "$1" == "CVS" ]; then
  # for things in CMSSW CVS
  ./getHLT.py $HLTtable $HLTid GEN-HLT
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDefaultOutput::outputCommands         --format python > HLTDefaultOutput_cff.py
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDefaultOutputWithFEDs::outputCommands --format python > HLTDefaultOutputWithFEDs_cff.py
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDebugOutput::outputCommands           --format python > HLTDebugOutput_cff.py
  edmConfigFromDB --configName $HLTcontent --nopaths --noes --nopsets --noservices --cff --blocks hltDebugWithAlCaOutput::outputCommands   --format python > HLTDebugWithAlCaOutput_cff.py

  ls -lt HLT*_cff.py
  mv -f HLT*_cff.py ../python
else
  # for things NOT in CMSSW CVS:
  ./getHLT.py $HLTtable $HLTid
fi
