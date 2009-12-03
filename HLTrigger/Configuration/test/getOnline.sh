#! /bin/bash

HLT='/online/beamhalo/week47/HLT'

rm -f OnLine_HLT_TEST.py

./getHLT.py --process TEST --full --offline --mc   "$HLT" TEST
mv OnLine_HLT_TEST.py offline_mc.py
./getHLT.py --process TEST --full --offline --data "$HLT" TEST
mv OnLine_HLT_TEST.py offline_data.py
./getHLT.py --process TEST --full --online  --data "$HLT" TEST
mv OnLine_HLT_TEST.py online_data.py
