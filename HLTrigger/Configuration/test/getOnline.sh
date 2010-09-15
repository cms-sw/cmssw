#! /bin/bash

HLT='/online/collisions/2010/week37/HLT'
L1T='L1Menu_Commissioning2010_v4'

hltGetConfiguration $HLT --process TEST --full --offline --mc             --unprescale --dataset '/RelValTTbar/CMSSW_3_6_3-START36_V10-v1/GEN-SIM-DIGI-RAW-HLTDEBUG' > offline_mc.py
hltGetConfiguration $HLT --process TEST --full --offline --data --l1 $L1T --unprescale > offline_data.py
hltGetConfiguration $HLT --process TEST --full --online  --data --l1 $L1T --unprescale > online_data.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
