#! /bin/bash

HLT='/online/collisions/2011/ParadiseLost/v3/HLT'
L1T='L1GtTriggerMenu_L1Menu_Collisions2011_v0a_mc'

hltGetConfiguration $HLT --process TEST --full --offline --mc   --unprescale > offline_mc.py
hltGetConfiguration $HLT --process TEST --full --offline --data --unprescale > offline_data.py
hltGetConfiguration $HLT --process TEST --full --online  --data --unprescale > online_data.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
