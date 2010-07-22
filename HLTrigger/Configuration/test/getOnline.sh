#! /bin/bash

HLT='/online/collisions/2010/week28/HLT'
# L1T='L1Menu_Commissioning2010_v3'

rm -f OnData_HLT_TEST.py
rm -f OnLine_HLT_TEST.py

./getHLT.py --process TEST --full --offline --mc   --unprescale $HLT TEST
mv OnLine_HLT_TEST.py offline_mc.py
./getHLT.py --process TEST --full --offline --data --unprescale $HLT TEST
mv OnData_HLT_TEST.py offline_data.py
./getHLT.py --process TEST --full --online  --data --unprescale $HLT TEST
mv OnData_HLT_TEST.py online_data.py

{
  TABLE=$(echo $HLT | cut -d: -f2)
  DB=$(echo $HLT | cut -d: -f1 -s)
  true ${DB:=hltdev}

  edmConfigFromDB --$DB --configName $TABLE | hltDumpStream 
} > streams.txt
