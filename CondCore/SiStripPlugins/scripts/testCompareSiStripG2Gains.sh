#!/bin/bash -e

display_usage() { 
    echo "This script must be run giving the following arguments." 
    echo -e "./testCompare.sh <name of tag> <first IOV> <last IOV> <sqlite file> \n\n"
    echo -e "example: \n ./testCompare.sh SiStripApvGain_FromParticles_GR10_v1_express 300577 302322 toCompare.db \n" 
} 

# if less than two arguments supplied, display usage 
if [  $# -le 3 ]; then
    display_usage
    exit 1
fi
 
# check whether user had supplied -h or --help . If yes display usage 
if [[ ( "$1" == "--help" ) ||  "$1" == "-h" ]]; then
    display_usage
    exit 0
fi

# Save current working dir so img can be outputted there later
W_DIR=$(pwd)

STARTIOV=$2
ENDIOV=$3

# Check if ENDIOV is greater than or equal to STARTIOV
if (( $ENDIOV < $STARTIOV )); then
    echo "Error: ENDIOV ($ENDIOV) is less than STARTIOV ($STARTIOV). Skipping comparisons"
    exit 0
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
eval "$(scram run -sh)"

# Go back to original working directory
cd "$W_DIR"

plotTypes=(
    SiStripApvGainsComparatorSingleTag
    SiStripApvGainsValuesComparatorSingleTag
    SiStripApvGainsComparatorByRegionSingleTag
    SiStripApvGainsRatioComparatorByRegionSingleTag
    SiStripApvGainByPartition
)

mkdir -p "$W_DIR/results_$2-$3"

# Remove any existing PNG files
if ls *.png 1> /dev/null 2>&1; then
    rm *.png
fi

for i in "${plotTypes[@]}"; do
    echo "Making plot ${i}"

    # Run get payload data script
    getPayloadData.py \
        --plugin pluginSiStripApvGain_PayloadInspector \
        --plot plot_${i} \
        --tag "$1" \
        --time_type Run \
        --iovs '{"start_iov": "'$STARTIOV'", "end_iov": "'$ENDIOV'"}' \
        --db sqlite_file:"$4" \
        --test

    # Check if the command failed
    if [ $? -ne 0 ]; then
        echo "Error in getPayloadData for plot ${i}, exiting..."
        exit 1
    fi

    mv *.png "$W_DIR/results_$2-$3/${i}_$1_$2-$3.png"
done

plotTypes2=(
    SiStripApvGainCompareByPartitionTwoTags
    SiStripApvGainDiffByPartitionTwoTags
)

for i in "${plotTypes2[@]}"; do
    echo "Making plot ${i}"

    # Run get payload data script
    getPayloadData.py \
        --plugin pluginSiStripApvGain_PayloadInspector \
        --plot plot_${i} \
        --tag "$1" \
        --tagtwo "$1" \
        --time_type Run \
        --iovs '{"start_iov": "'$STARTIOV'", "end_iov": "'$STARTIOV'"}' \
        --iovstwo '{"start_iov": "'$ENDIOV'", "end_iov": "'$ENDIOV'"}' \
        --db sqlite_file:"$4" \
        --test

    # Check if the command failed
    if [ $? -ne 0 ]; then
        echo "Error in getPayloadData for plot ${i}, exiting..."
        exit 1
    fi

    mv *.png "$W_DIR/results_$2-$3/${i}_$1_$2-$3.png"
done

plotTypes3=(
    SiStripApvGainsAvgDeviationRatioWithPreviousIOVTrackerMap
    SiStripApvGainsMaxDeviationRatioWithPreviousIOVTrackerMap
)

for i in "${plotTypes3[@]}"
do
    for j in 1 2 3
    do
	echo "Making plot ${i} with ${j} sigmas"
	# Run get payload data script
	getPayloadData.py \
	    --plugin pluginSiStripApvGain_PayloadInspector \
	    --plot plot_${i} \
	    --tag $1 \
	    --time_type Run \
	    --iovs  '{"start_iov": "'$STARTIOV'", "end_iov": "'$ENDIOV'"}' \
	    --input_params '{"nsigma":"'${j}'"}' \
	    --db sqlite_file:$4 \
	    --test

	# Check if the command failed
	if [ $? -ne 0 ]; then
            echo "Error in getPayloadData for plot ${i}, exiting..."
            exit 1
	fi

	mv *.png "$W_DIR/results_$2-$3/${i}_${j}sigmas_$1_$2-$3.png"
    done
done
