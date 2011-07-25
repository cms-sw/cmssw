#! /bin/bash

HLT='/online/collisions/2011/2e33/v1.2/HLT'
L1T='L1GtTriggerMenu_L1Menu_Collisions2011_v5_mc'

# tests
hltGetConfiguration $HLT --process TEST                               --full --online  --data --unprescale --l1 $L1T --l1-emulator > online_data.py
hltGetConfiguration $HLT --process TEST                               --full --offline --data --unprescale --l1 $L1T --l1-emulator > offline_data.py
hltGetConfiguration $HLT --process TEST    --globaltag auto:startup   --full --offline --mc   --unprescale --l1 $L1T --l1-emulator > offline_mc.py

# standard 'cff' dumps - in CVS
hltGetConfiguration $HLT                                              --cff  --offline --data                                      > ../python/HLT_GRun_data_cff.py
hltGetConfiguration $HLT                                              --cff  --offline --mc                                        > ../python/HLT_GRun_cff.py

# standard 'cfg' dumps - in CVS
hltGetConfiguration $HLT --process HLTGRun --globaltag auto:hltonline --full --offline --data --unprescale --l1 $L1T --l1-emulator > OnData_HLT_GRun.py
hltGetConfiguration $HLT --process HLTGRun --globaltag auto:startup   --full --offline --mc   --unprescale --l1 $L1T --l1-emulator > OnLine_HLT_GRun.py 

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
