#! /bin/bash

HLT='/online/collisions/2010/week21/HLT'
L1T='L1Menu_Commissioning2010_v2'

rm -f OnData_HLT_TEST.py
rm -f OnLine_HLT_TEST.py

./getHLT.py --process TEST --full --offline --mc   --l1 $L1T $HLT TEST
mv OnLine_HLT_TEST.py offline_mc.py
./getHLT.py --process TEST --full --offline --data --l1 $L1T $HLT TEST
mv OnData_HLT_TEST.py offline_data.py
./getHLT.py --process TEST --full --online  --data --l1 $L1T $HLT TEST
mv OnData_HLT_TEST.py online_data.py

{
  head -n1 online_data.py
  echo
  edmConfigFromDB --configName $HLT | hltDumpStream 
} > streams.txt
