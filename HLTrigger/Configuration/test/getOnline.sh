#! /bin/bash

HLT='/online/collisions/2012/8e33/v2.2/HLT'
L1T='L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc'
#L1T='L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2012_v3/sqlFile/L1Menu_Collisions2012_v3_mc.db'

# tests
hltGetConfiguration $HLT --process TEST                                  --full --online  --data --unprescale --l1 $L1T --l1-emulator > online_data.py
hltGetConfiguration $HLT --process TEST                                  --full --offline --data --unprescale --l1 $L1T --l1-emulator > offline_data.py
hltGetConfiguration $HLT --process TEST    --globaltag auto:startup_GRun --full --offline --mc   --unprescale --l1 $L1T --l1-emulator > offline_mc.py

# standard 'cff' dumps - in CVS
hltGetConfiguration $HLT                                              --cff  --offline --data                                      > ../python/HLT_GRun_data_cff.py
hltGetConfiguration $HLT                                              --cff  --offline --mc                                        > ../python/HLT_GRun_cff.py
diff -C0 ../python/HLT_GRun_data_cff.py ../python/HLT_GRun_cff.py

# FastSim 'cff' dump - in CVS
hltGetConfiguration $HLT                                              --fastsim                                                    > ../python/HLT_GRun_Famos_cff.py

# standard 'cfg' dumps - in CVS
hltGetConfiguration $HLT --process HLTGRun --globaltag auto:hltonline_GRun --full --offline --data --unprescale --l1 $L1T          > OnData_HLT_GRun.py
hltGetConfiguration $HLT --process HLTGRun --globaltag auto:startup_GRun   --full --offline --mc   --unprescale --l1 $L1T          > OnLine_HLT_GRun.py 


{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
