#!/bin/sh

BaseDir=/home/cmstacuser/historyDQM/Cron/Scripts

########################################################
# THIS MUST BE SET OTHERWISE THE CRON WILL NOT HAVE IT #
########################################################
CMS_PATH=/afs/cern.ch/cms

# Define the release version to use
CMSSW_version=/home/cmstacuser/historyDQM/CMSSW_Releases/CMSSW_3_2_5
source /afs/cern.ch/cms/sw/cmsset_default.sh
echo $CMSSW_version
cd ${CMSSW_version}/src
eval `scramv1 r -sh`

source ${BaseDir}/SubDetScript.sh
source ${BaseDir}/UpdatePlots.sh

# Start an endless loop
while [ 1 = 1 ]
do
    # BaseDir=/home/cmstacuser/historyDQM/Cron/Scripts
    cd ${BaseDir}

    # -------------------- #
    # IMPORTANT PARAMETERS #
    # -------------------- #

    AuthenticationPath="/home/cmstacuser/historyDQM/Cron/Scripts/Authentication"
    # Database="oracle://cms_orcoff_prep/CMS_COND_STRIP"
    Database="oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE"
    # Database="sqlite_file:dbfile.db"

    # The final tag name will be: TagName_DetName_TagVersion: e.g. HDQM_SiStrip_V1
    TagName="HDQM"

    # The first run from which to start populating the database
    FirstRun=108239
    # The last run for which to populate the database (-1 = all runs)
    LastRun=-1

    SubDets=(     "SiStrip" "SiPixel" "Tracking" "RPC")
    TagVersions=( "V3"      "V3"      "V1"       "V1")

    # -------------------- #
    # -------------------- #

    cd ${BaseDir}

    Update=0

    k=0
    SubDetNumber=${#SubDets[*]}

    while [ "$k" -lt "$SubDetNumber" ]
	do
	SubDet=${SubDets[k]}
	TagVersion=${TagVersions[k]}
	echo "Processing new runs for $SubDet"

	# echo
	# echo ${SubDet}
	# echo ${TagName}_${SubDet}_${TagVersion}
	# echo ${Database}
	# echo ${AuthenticationPath}
	# echo ${FirstRun}
	# echo ${LastRun}
	# echo ${CMSSW_version}
	# echo

	SubDetScript ${SubDet} ${TagName}_${SubDet}_${TagVersion} ${Database} ${AuthenticationPath} ${FirstRun} ${LastRun} ${CMSSW_version} ${BaseDir}/Templates
	let "Update+=$?"

	let "k+=1"

	TagNames[$k]=${TagName}_${SubDet}_${TagVersion}
    done

    # Something has changed. Update all the histograms.
    if [ ${Update} -ge 1 ]; then
	# This is needed to pass the array
	argument1=`echo ${TagNames[@]}`
	argument2=`echo ${SubDets[@]}`
	UpdatePlots "${argument1}" "${argument2}" ${CMSSW_version}
    else
	echo "Nothing changed, not redoing the plots."
    fi


    # argument1=`echo ${TagNames[@]}`
    # argument2=`echo ${SubDets[@]}`
    # UpdatePlots "${argument1}" "${argument2}" ${CMSSW_version}


    # Wait one hour
    sleep 3600
done
