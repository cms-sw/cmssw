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
# Test Pedestals
####################
getPayloadData.py \
    --plugin pluginSiStripPedestals_PayloadInspector \
    --plot plot_SiStripPedestalsTest \
    --tag SiStripPedestals_v2_prompt \
    --time_type Run \
    --iovs '{"start_iov": "303420", "end_iov": "303420"}' \
    --db Prod \
    --test;

estimators=(Mean Min Max RMS)
plotTypes=(Strip APV Module)

mkdir -p $W_DIR/results

if [ -f *.png ]; then    
    rm *.png
fi

for i in "${estimators[@]}" 
do

    #// TrackerMaps

    getPayloadData.py \
	--plugin pluginSiStripPedestals_PayloadInspector \
	--plot plot_SiStripPedestals${i}_TrackerMap \
	--tag SiStripPedestals_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "303420"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripPedestals${i}_TrackerMap.png
    
    #// Summaries

    getPayloadData.py \
	--plugin pluginSiStripPedestals_PayloadInspector \
	--plot plot_SiStripPedestals${i}ByRegion \
	--tag SiStripPedestals_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "303420"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripPedestals${i}ByRegion.png

done

for j in "${plotTypes[@]}"
do  
    getPayloadData.py \
	--plugin pluginSiStripPedestals_PayloadInspector \
	--plot plot_SiStripPedestalValuePer${j} \
	--tag SiStripPedestals_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "303420"}' \
	--db Prod \
	--test ;
	
    mv *.png $W_DIR/results/SiStripPedestalsPer${j}Values.png

    getPayloadData.py \
	--plugin pluginSiStripPedestals_PayloadInspector \
	--plot plot_SiStripPedestalValueComparisonPer${j} \
	--tag SiStripPedestals_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "313120"}' \
	--db Prod \
	--test ;

    mv *.png $W_DIR/results/SiStripPedestalsPer${j}Comparison.png

done
