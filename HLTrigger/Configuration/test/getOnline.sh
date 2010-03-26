#! /bin/bash

HLT='/online/collisions/2010/week13/HLT'
L1T='L1GtTriggerMenu_L1Menu_Commissioning2010_v0_mc'

rm -f OnData_HLT_TEST.py
rm -f OnLine_HLT_TEST.py

./getHLT.py --process TEST --full --offline --mc   $HLT --l1 $L1T TEST
mv OnLine_HLT_TEST.py offline_mc.py
./getHLT.py --process TEST --full --offline --data $HLT --l1 $L1T TEST
mv OnData_HLT_TEST.py offline_data.py
./getHLT.py --process TEST --full --online  --data $HLT --l1 $L1T TEST
mv OnData_HLT_TEST.py online_data.py

{
  head -n1 online_data.py
  echo
  edmConfigFromDB --configName $HLT | hltDumpStream 
} > streams.txt
