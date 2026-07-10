#!/bin/bash

# Test suite for miniaod from prompt

# Pass in name and status
declare -a arr=("ppEra_Run3_2025" "ppEra_Run3_2026" "ppEra_Run3_2026_UPC" "ppEra_Run3_pp_on_PbPb_approxSiStripClusters_2026" "ppEra_Run3_pp_on_PbPb_2026")
for scenario in "${arr[@]}"
do
    rm -rf RunPromptRecoCfg.pkl
    python3 ${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --miniaod --global-tag GLOBALTAG --lfn=/store/whatever
    if [ ! -f "RunPromptRecoCfg.pkl" ]; then
	echo "Can't dump RunPromptRecoCfg.pkl"
	exit 1;
    fi
 	
python3 <<EOF
import sys, pickle 
f=open('RunPromptRecoCfg.pkl','rb')
process = pickle.load(f);
if not hasattr(process, "write_MINIAOD"):
   sys.exit(1)

mod = process.write_MINIAOD
if not hasattr(mod, "overrideBranchesSplitLevel"):
   sys.exit(1)
EOF

    rc=$?
    if [ $rc -ne 0 ]; then
	echo "check process.write_MINIAOD"
	exit 1;
    fi

    echo '>>>> Done! <<<<'
done
