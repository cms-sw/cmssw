#! /bin/bash

# The script is used to query file list from DBS
# use ./getrun.sh -h for help on the usage
#
#                           Dayong WANG June, 2009

function usage {

    echo "Usage: "
    echo "    1) query file list with run number and dataset name: ./getrun.sh -r RUNNUMBER (datasetname)"
    echo "    2) query file list with run number range [MIN, MAX) and dataset name:  ./getrun.sh  RUNNUMBER_MIN  RUNNUMBER_MAX (datasetname)"

}


while getopts ":r:h" opt; do
    case $opt in

	r)
	    if [ $# -eq 2 ]; then
		mydataset="/Cosmics/Commissioning08_CRAFT_ALL_V9_225_ReReco_FromTrackerPointing_v1/RECO"
	    elif [ $# -eq 3 ]; then
		mydataset=$3
	    elif [ $# -eq 0 ]; then
		echo "please provide the run number and (data set name) to query!"
		exit 2
	    fi

	    ./DDSearchCLI.py --verbose=0 --limit=-1 --input="find file where run=$OPTARG and dataset=$mydataset" | grep "root"
	    ;;
	\?)
	    echo "Invalid option: -$OPTARG" >&2
	    exit 1
	    ;;
	h)
	    usage
	    ;;
	:)
	    echo "Option -$OPTARG requires an argument to specify run number query  mode" >&2
	    exit 1
	    ;;
    esac
    exit 0
done

if [ $# -eq 2 ]; then
    mydataset="/Cosmics/Commissioning08_CRAFT_ALL_V9_225_ReReco_FromTrackerPointing_v1/RECO"
elif [ $# -eq 3 ]; then
    mydataset=$3
elif [ $# -lt 2 ]; then
    echo "please provide the range of run numbers and (data set name) to query!"
    exit 2
fi

./DDSearchCLI.py --verbose=0 --limit=-1 --input="find file where run >= $1 and run < $2 and dataset=$mydataset" | grep "root"





