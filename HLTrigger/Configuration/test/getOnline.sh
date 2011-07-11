#! /bin/bash

HLT='/online/collisions/2011/2e33/v1.0/HLT/V2'
L1T='L1GtTriggerMenu_L1Menu_Collisions2011_v4_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2011_v4/sqlFile/L1Menu_Collisions2011_v4_mc.db'

hltGetConfiguration $HLT --process TEST --globaltag auto:startup   --full --offline --mc   --unprescale > offline_mc.py
hltGetConfiguration $HLT --process TEST                            --full --offline --data --unprescale > offline_data.py
hltGetConfiguration $HLT --process TEST                            --full --online  --data --unprescale > online_data.py

hltGetConfiguration $HLT --cff --offline --mc   > ../python/HLT_GRun_cff.py
hltGetConfiguration $HLT --cff --offline --data > ../python/HLT_GRun_data_cff.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
