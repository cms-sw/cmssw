#! /bin/bash

L1T='L1GtTriggerMenu_STARTUP_v6'
HLT='/online/collisions/week49/HLT'

rm -f OnLine_HLT_TEST.py

./getHLT.py --l1 $L1T --process TEST --full --offline --mc   $HLT TEST
mv OnLine_HLT_TEST.py offline_mc.py
./getHLT.py --l1 $L1T --process TEST --full --offline --data $HLT TEST
mv OnLine_HLT_TEST.py offline_data.py
./getHLT.py --l1 $L1T --process TEST --full --online  --data $HLT TEST
mv OnLine_HLT_TEST.py online_data.py
