#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script

mkdir -p $W_DIR/results

if [ -f *.png ]; then    
    rm *.png
fi

echo "Testing the Luminosity Corrections Summary plot"

getPayloadData.py \
    --plugin pluginLumiCorrections_PayloadInspector \
    --plot plot_LumiCorrectionsSummary \
    --tag LumiPCC_Corrections_prompt \
    --time_type Lumi \
    --iovs '{"start_iov": "1545372182773899", "end_iov": "1545372182773899"}' \
    --db Prod \
    --test ;

mv *.png  $W_DIR/results/LuminosityCorrectionsSummary.png
