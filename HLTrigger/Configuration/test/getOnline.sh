#! /bin/bash

# load common HLT functions
if [ -f "$CMSSW_BASE/src/HLTrigger/Configuration/common/utils.sh" ]; then
  source "$CMSSW_BASE/src/HLTrigger/Configuration/common/utils.sh"
elif [ -f "$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/common/utils.sh" ]; then
  source "$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/common/utils.sh"
else
  exit 1
fi

HLT='/online/collisions/2012/8e33/v2.2/HLT'
L1T='L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc'
#L1T='L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,sqlite_file:/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2012_v3/sqlFile/L1Menu_Collisions2012_v3_mc.db'

# tests
hltGetConfiguration $HLT --process TEST                                    --full --data --unprescale --l1 $L1T --l1-emulator > online_data.py
hltGetConfiguration $HLT --process TEST                                    --full --data --unprescale --l1 $L1T --l1-emulator > offline_data.py
hltGetConfiguration $HLT --process TEST    --globaltag auto:startup_GRun   --full --mc   --unprescale --l1 $L1T --l1-emulator > offline_mc.py

# standard 'cff' dumps - in CVS
hltGetConfiguration $HLT                                                   --cff  --data                                      > ../python/HLT_GRun_data_cff.py
hltGetConfiguration $HLT                                                   --cff  --mc                                        > ../python/HLT_GRun_cff.py
diff -C0 ../python/HLT_GRun_data_cff.py ../python/HLT_GRun_cff.py

# standard 'cfg' dumps - in CVS
hltGetConfiguration $HLT --process HLTGRun --globaltag auto:hltonline_GRun --full --data --unprescale --l1 $L1T               > OnData_HLT_GRun.py
hltGetConfiguration $HLT --process HLTGRun --globaltag auto:startup_GRun   --full --mc   --unprescale --l1 $L1T               > OnLine_HLT_GRun.py

# dump streams, datasets and paths
read Vx DB TABLE <<< $(parse_HLT_menu "$HLT")
hltConfigFromDB --$Vx --$DB --configName $TABLE | hltDumpStream > streams.txt
