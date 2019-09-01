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

mkdir -p $W_DIR/results
declare -a arr=("325662" "325702" "325715" "325717")

for i in "${arr[@]}"
do
    echo $i
    getPayloadData.py \
	--plugin pluginSiStripFedCabling_PayloadInspector \
	--plot plot_SiStripFedCabling_TrackerMap \
	--tag SiStripFedCabling_GR10_v1_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripFedCabling_${i}_TrackerMap.png
  

      getPayloadData.py \
	--plugin pluginSiStripFedCabling_PayloadInspector \
	--plot plot_SiStripFedCabling_Summary \
	--tag SiStripFedCabling_GR10_v1_hlt \
	--time_type Run \
	--iovs '{"start_iov": "'$i'", "end_iov": "'$i'"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results/SiStripFedCabling_${i}_Summary.png
done
