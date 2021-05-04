#!/bin/bash

# Save current working dir so img can be outputted there later
W_DIR=$(pwd);

# Set SCRAM architecture var
SCRAM_ARCH=slc6_amd64_gcc530; 
export SCRAM_ARCH;

cd $W_DIR;
source /afs/cern.ch/cms/cmsset_default.sh;
eval `scram run -sh`;

mkdir -p $W_DIR/results_alignments

#*************************************************************************#
elements=(X Y Z Alpha Beta Gamma)

for i in "${elements[@]}"
do
    echo "Processing: $i coordinate"

    getPayloadData.py  \
	--plugin pluginTrackerAlignment_PayloadInspector \
	--plot plot_TrackerAlignmentCompare${i} \
	--tag TrackerAlignment_PCL_byRun_v2_express \
	--time_type Run \
	--iovs '{"start_iov": "303809", "end_iov": "303886"}' \
	--db Prod \
	--test;

    mv *.png $W_DIR/results_alignments/TrackerAlignmentCompare${i}.png

done

#*************************************************************************#
elements=(BPix FPix TIB TOB TID TEC)

for i in "${elements[@]}"
do
    echo "Processing: $i partition"
    
    getPayloadData.py  \
 	--plugin pluginTrackerAlignment_PayloadInspector \
 	--plot plot_TrackerAlignmentSummary${i} \
 	--tag TrackerAlignment_PCL_byRun_v2_express \
 	--time_type Run \
	--iovs '{"start_iov": "303809", "end_iov": "303886"}' \
  	--db Prod \
 	--test;
    
    mv *.png $W_DIR/results_alignments/TrackerAlignmentSummary${i}.png
done

#*************************************************************************#
elements=(X Y Z)

for i in "${elements[@]}"
do
    echo "Processing: $i partition"
    
    getPayloadData.py  \
 	--plugin pluginTrackerAlignment_PayloadInspector \
 	--plot plot_${i}_BPixBarycenterHistory \
 	--tag TrackerAlignment_v21_offline\
 	--time_type Run \
	--iovs '{"start_iov": "294034", "end_iov": "305898"}' \
  	--db Prod \
 	--test;
    
done

# add example of single IOV barycenter dump
#*************************************************************************#
getPayloadData.py \
    --plugin pluginTrackerAlignment_PayloadInspector \
    --plot plot_TrackerAlignmentBarycenters \
    --tag TrackerAlignment_Upgrade2017_realistic_v3 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;


# add examples of Pixel barycenter comparison
#*************************************************************************#
getPayloadData.py \
    --plugin pluginTrackerAlignment_PayloadInspector \
    --plot plot_PixelBarycentersCompare \
    --tag TrackerAlignment_v28_offline \
    --time_type Run \
    --iovs '{"start_iov": "250000", "end_iov": "300000"}' \
    --db Prod \
    --test ;

getPayloadData.py \
    --plugin pluginTrackerAlignment_PayloadInspector \
    --plot plot_PixelBarycentersCompareTwoTags \
    --tag TrackerAlignment_Ideal62X_mc \
    --tagtwo TrackerAlignment_Upgrade2017_design_v4 \
    --time_type Run \
    --iovs '{"start_iov": "1", "end_iov": "1"}' \
    --iovstwo '{"start_iov": "1", "end_iov": "1"}' \
    --db Prod \
    --test ;
