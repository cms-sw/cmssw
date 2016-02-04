#! /bin/bash

HLT='/online/collisions/2011/5e32/v6.0/HLT'
L1T='L1GtTriggerMenu_L1Menu_Collisions2011_v1_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2011_v1/sqlFile/L1Menu_Collisions2011_v1_mc.db'

hltGetConfiguration $HLT --process TEST --l1 $L1T --full --offline --mc   --unprescale > offline_mc.py
hltGetConfiguration $HLT --process TEST --l1 $L1T --full --offline --data --unprescale > offline_data.py
hltGetConfiguration $HLT --process TEST --l1 $L1T --full --online  --data --unprescale > online_data.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
