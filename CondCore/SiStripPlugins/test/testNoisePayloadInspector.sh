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

####################
# Test Noise
####################
/afs/cern.ch/user/c/condbpro/public/BROWSER_PI/getPayloadData.py \
    --plugin pluginSiStripNoises_PayloadInspector \
    --plot plot_SiStripNoisesTest \
    --tag SiStripNoise_v2_prompt \
    --time_type Run \
    --iovs '{"start_iov": "303420", "end_iov": "303420"}' \
    --db Prod \
    --test;

estimators=(Mean Min Max RMS)

mkdir -p $W_DIR/results

if [ -f *.png ]
then    
    rm *.png
fi

for i in "${estimators[@]}" 
do

    #// TrackerMaps

    /afs/cern.ch/user/c/condbpro/public/BROWSER_PI/getPayloadData.py \
	--plugin pluginSiStripNoises_PayloadInspector \
	--plot plot_SiStripNoise${i}_TrackerMap \
	--tag SiStripNoise_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "303420"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripNoises${i}_TrackerMap.png

    #// Summaries

    /afs/cern.ch/user/c/condbpro/public/BROWSER_PI/getPayloadData.py \
	--plugin pluginSiStripNoises_PayloadInspector \
	--plot plot_SiStripNoise${i}ByPartition \
	--tag SiStripNoise_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "303420"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripNoises${i}ByPartition.png

done

