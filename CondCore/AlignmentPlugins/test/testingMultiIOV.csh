#!/bin/tcsh

# Save current working dir so img can be outputted there later
set W_DIR = $PWD;

cd $W_DIR;
source /afs/cern.ch/cms/cmsset_default.csh;
eval `scram run -csh`;

mkdir -p $W_DIR/results_MultiIOV

set IOVs = (290550 292200 294604 295380 295702 297104 297417 298757 298809 298901 299228 299685 300662 302539 303796 303809 303818 303886 303949 304210 304334 304366 304653 304779 304933 304972 305047 305060 305311 305767 305782 305834 305841 305843 305899 305979 306037 306049 306052 306053 306088 306092 306093 306139 306460 306488 306532 306658 306827 306856 306930 306933 306937)
set last=$#IOVs

foreach i (`seq $#IOVs`) 
    
    if($i == $last) then
	continue
    endif

    @ j = $i + 1
    echo "Processing: $IOVs[$i] vs $IOVs[$j]"

    getPayloadData.py  \
    	--plugin pluginTrackerAlignment_PayloadInspector \
    	--plot plot_TrackerAlignmentCompareX \
    	--tag TrackerAlignment_PCL_byRun_v2_express \
    	--time_type Run \
    	--iovs '{"start_iov": "'$IOVs[$i]'", "end_iov": "'$IOVs[$j]'"}' \
    	--db Prod \
    	--test;

    mv *.png $W_DIR/results_MultiIOV/compareX_$IOVs[$i]_vs$IOVs[$j].png

end
