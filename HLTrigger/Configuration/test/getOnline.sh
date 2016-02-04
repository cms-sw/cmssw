#! /bin/bash

HLT='/online/collisions/2011/5e32/v8.1/HLT'
L1T='L1GtTriggerMenu_L1Menu_Collisions2011_v1_mc,frontier://FrontierProd/CMS_COND_31X_L1T'

hltGetConfiguration $HLT --process TEST --globaltag auto:startup   --full --offline --mc   --unprescale > offline_mc.py
hltGetConfiguration $HLT --process TEST --globaltag auto:hltonline --full --offline --data --unprescale > offline_data.py
hltGetConfiguration $HLT --process TEST                            --full --online  --data --unprescale > online_data.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
