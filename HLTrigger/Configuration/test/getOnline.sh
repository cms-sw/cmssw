#! /bin/bash

HLT='/online/collisions/2010/week44/HLT'
L1T='sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/sqlFile/L1Menu_CollisionsHeavyIons2010_v2_mc.db:L1Menu_CollisionsHeavyIons2010_v2'

hltGetConfiguration $HLT --process TEST --full --offline --mc   --l1 $L1T --globaltag START39_V4HI::All --unprescale > offline_mc.py
hltGetConfiguration $HLT --process TEST --full --offline --data --l1 $L1T                               --unprescale > offline_data.py
hltGetConfiguration $HLT --process TEST --full --online  --data --l1 $L1T                               --unprescale > online_data.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
