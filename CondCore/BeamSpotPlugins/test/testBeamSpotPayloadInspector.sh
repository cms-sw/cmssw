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

coordinates=(X Y Z SigmaX SigmaY SigmaZ dXdZ dYdZ)
types=(History RunHistory) #TimeHistory) not testable

mkdir -p $W_DIR/results

if [ -f *.png ]; then    
    rm *.png
fi

for i in "${types[@]}"
do
    for j in "${coordinates[@]}"
    do
	echo "plot_BeamSpot_${i}${j}"

	getPayloadData.py \
	    --plugin pluginBeamSpot_PayloadInspector \
	    --plot plot_BeamSpot_${i}${j} \
	    --tag BeamSpotObjects_PCL_byLumi_v0_prompt \
	    --time_type Lumi \
	    --iovs '{"start_iov": "1406876667346979", "end_iov": "1406876667347162"}' \
	    --db Prod \
	    --test;
    done
done
