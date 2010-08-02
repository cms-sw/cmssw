#! /bin/bash

HLT='/online/collisions/2010/week31/HLT'
L1T='L1Menu_Commissioning2010_v3'

rm -f OnData_HLT_TEST.py
rm -f OnLine_HLT_TEST.py

hltGetConfiguration $HLT --process TEST --full --offline --mc   --unprescale --l1 $L1T --dataset '/RelValTTbar/CMSSW_3_6_3-START36_V10-v1/GEN-SIM-DIGI-RAW-HLTDEBUG' > offline_mc.py
hltGetConfiguration $HLT --process TEST --full --offline --data --unprescale --l1 $L1T > offline_data.py
hltGetConfiguration $HLT --process TEST --full --online  --data --unprescale --l1 $L1T > online_data.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
