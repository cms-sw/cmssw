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

if [ ! -d $W_DIR/results ]; then
    mkdir $W_DIR/results
fi
 
# reference
getPayloadData.py \
    --plugin pluginSiStripNoises_PayloadInspector \
    --plot plot_SiStripNoiseValueComparisonPerAPV \
    --tag SiStripNoise_GR10_v1_hlt \
    --time_type Run \
    --iovs '{"start_iov": "312968", "end_iov": "313120"}' \
    --db Prod \
    --test;
	
mv *.png $W_DIR/results/SiStripNoisesPerAPVValues_ref.png

# target
getPayloadData.py \
    --plugin pluginSiStripNoises_PayloadInspector \
    --plot plot_SiStripNoiseConsistencyCheck \
    --tag SiStripNoise_GR10_v1_hlt \
    --time_type Run \
    --iovs '{"start_iov": "312968", "end_iov": "313120"}' \
    --db Prod \
    --test;
	
mv *.png $W_DIR/results/SiStripNoisesPerAPVValues_tar.png

# LA test
getPayloadData.py \
    --plugin pluginSiStripLorentzAngle_PayloadInspector \
    --plot plot_SiStripLorentzAngleTest \
    --tag SiStripLorentzAngleDeco_GR10_v1_prompt \
    --time_type Run \
    --iovs '{"start_iov": "132646", "end_iov": "132646"}' \
    --db Prod \
    --test;
	
mv *.png $W_DIR/results/SiStripLorentzAngleTest.png

getPayloadData.py \
    --plugin pluginSiStripLorentzAngle_PayloadInspector \
    --plot plot_SiStripLorentzAngleCompareByRegion \
    --tag SiStripLorentzAngleDeco_GR10_v1_prompt \
    --time_type Run \
    --iovs '{"start_iov": "132646", "end_iov": "254501"}' \
    --db Prod \
    --test;
	
mv *.png $W_DIR/results/SiStripLorentzAngleComparebyRegion.png
