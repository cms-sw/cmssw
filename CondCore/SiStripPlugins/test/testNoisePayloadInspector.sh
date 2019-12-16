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
getPayloadData.py \
    --plugin pluginSiStripNoises_PayloadInspector \
    --plot plot_SiStripNoisesTest \
    --tag SiStripNoise_v2_prompt \
    --time_type Run \
    --iovs '{"start_iov": "303420", "end_iov": "303420"}' \
    --db Prod \
    --test;

####################
# Single DetId
####################
getPayloadData.py \
    --plugin pluginSiStripNoises_PayloadInspector \
    --plot plot_SiStripNoiseValuePerDetId \
    --tag SiStripNoise_v2_prompt \
    --time_type Run \
    --iovs '{"start_iov": "303420", "end_iov": "303420"}' \
    --db Prod \
    --input_params '{"DetId":"470065830"}' \
    --test ;

estimators=(Mean Min Max RMS)
plotTypes=(Strip APV Module)
partition=(TIB TOB TEC TID)

mkdir -p $W_DIR/results

if [ -f *.png ]; then    
    rm *.png
fi

for i in "${estimators[@]}" 
do

    #// TrackerMaps

    getPayloadData.py \
	--plugin pluginSiStripNoises_PayloadInspector \
	--plot plot_SiStripNoise${i}_TrackerMap \
	--tag SiStripNoise_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "303420"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripNoises${i}_TrackerMap.png

    #// Summaries

    getPayloadData.py \
	--plugin pluginSiStripNoises_PayloadInspector \
	--plot plot_SiStripNoise${i}ByRegion \
	--tag SiStripNoise_v2_prompt \
	--time_type Run \
	--iovs '{"start_iov": "303420", "end_iov": "303420"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripNoises${i}ByRegion.png

done

for j in "${plotTypes[@]}"
do  
    getPayloadData.py \
	--plugin pluginSiStripNoises_PayloadInspector \
	--plot plot_SiStripNoiseValuePer${j} \
	--tag SiStripNoise_GR10_v1_hlt \
	--time_type Run \
	--iovs  '{"start_iov": "313210", "end_iov": "313120"}' \
	--db Prod \
	--test;
	
    mv *.png $W_DIR/results/SiStripNoisesPer${j}Values.png

    getPayloadData.py \
	--plugin pluginSiStripNoises_PayloadInspector \
	--plot plot_SiStripNoiseValueComparisonPer${j}SingleTag \
	--tag SiStripNoise_GR10_v1_hlt \
	--time_type Run \
	--iovs '{"start_iov": "312968", "end_iov": "313120"}' \
	--db Prod \
	--test ;

    mv *.png $W_DIR/results/SiStripNoisesPer${j}ComparisonSingleTag.png

done

for j in "${partition[@]}"
do
       getPayloadData.py \
        --plugin pluginSiStripNoises_PayloadInspector \
	--plot plot_${j}NoiseLayerRunHistory \
        --tag SiStripNoise_GR10_v1_hlt \
	--time_type Run \
        --iovs  '{"start_iov": "315000", "end_iov": "325000"}' \
	--db Prod \
        --test;

    mv *.png $W_DIR/results/${j}NoiseLayerRunHistory.png
done
