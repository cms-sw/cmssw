#!/bin/bash
# Save current working dir so img can be outputted there later
W_DIR=$(pwd);
# Set SCRAM architecture var
SCRAM_ARCH=slc7_amd64_gcc900;
export SCRAM_ARCH;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;
# Go back to original working directory
cd $W_DIR;
# Run get payload data script
if [ -d $W_DIR/plots_VCal ]; then
    rm -fr $W_DIR/plots_VCal
fi

mkdir $W_DIR/plots_VCal

TAGA=SiPixelVCal_v1
TAGB=SiPixelVCal_phase1_2021_v0

## start with single tag plots

singleTagPlots=(SiPixelVCalValues SiPixelVCalSlopeValuesBarrel SiPixelVCalSlopeValuesEndcap SiPixelVCalOffsetValuesBarrel SiPixelVCalOffsetValuesEndcap)

for i in "${singleTagPlots[@]}"
do
    echo "Processing: $i plot"

    getPayloadData.py  \
	--plugin pluginSiPixelVCal_PayloadInspector \
	--plot plot_${i} \
	--tag $TAGA \
	--time_type Run \
	--iovs '{"start_iov": "1", "end_iov": "1"}' \
	--db Prep \
	--test;

    mv *.png $W_DIR/plots_VCal/${i}.png
done

twoTagPlots=(SiPixelVCalSlopesBarrelCompareTwoTags SiPixelVCalOffsetsBarrelCompareTwoTags SiPixelVCalSlopesEndcapCompareTwoTags SiPixelVCalOffsetsEndcapCompareTwoTags SiPixelVCalSlopesComparisonTwoTags SiPixelVCalOffsetsComparisonTwoTags) 

for j in "${twoTagPlots[@]}"
do
    echo "Processing: $j plot"

    getPayloadData.py  \
	--plugin pluginSiPixelVCal_PayloadInspector \
	--plot plot_${j} \
	--tag $TAGA \
	--tagtwo $TAGB \
	--time_type Run \
	--iovs '{"start_iov": "1", "end_iov": "1"}' \
	--iovstwo '{"start_iov": "1", "end_iov": "1"}' \
	--db Prep \
	--test;

    mv *.png $W_DIR/plots_VCal/${j}.png
done
