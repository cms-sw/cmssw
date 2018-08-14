#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc630;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script

if [ -d $W_DIR/plots ]; then
    rm -fr $W_DIR/plots
fi

mkdir $W_DIR/plots

####################
# Test RunInfo
####################
getPayloadData.py --plugin pluginRunInfo_PayloadInspector --plot plot_RunInfoParameters --tag runinfo_31X_hlt --time_type Run --iovs '{"start_iov": "311957", "end_iov": "311957"}' --db Prod --test ;
mv *.png $W_DIR/plots/wrong_payload.png

getPayloadData.py --plugin pluginRunInfo_PayloadInspector --plot plot_RunInfoParameters --tag runinfo_31X_hlt --time_type Run --iovs '{"start_iov": "312257", "end_iov": "312257"}' --db Prod --test ;
mv *.png $W_DIR/plots/fake_payload.png

getPayloadData.py --plugin pluginRunInfo_PayloadInspector --plot plot_RunInfoParameters --tag runinfo_31X_hlt --time_type Run --iovs '{"start_iov": "312259", "end_iov": "312259"}' --db Prod --test ;
mv *.png $W_DIR/plots/OK_payload.png