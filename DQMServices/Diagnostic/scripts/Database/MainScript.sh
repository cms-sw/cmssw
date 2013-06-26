#!/bin/sh

BaseDir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts

########################################################
# THIS MUST BE SET OTHERWISE THE CRON WILL NOT HAVE IT #
########################################################
CMS_PATH=/afs/cern.ch/cms

# Define the release version to use

# Use this from slc4
# CMSSW_version=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/CMSSW_Releases/CMSSW_3_2_6

# Use this from slc5
CMSSW_version=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/CMSSW_Releases/slc5/CMSSW_3_3_6
TemplatesDir=${CMSSW_version}/src/DQMServices/Diagnostic/scripts/Database/Templates

source /afs/cern.ch/cms/sw/cmsset_default.sh
echo $CMSSW_version
cd ${CMSSW_version}/src
eval `scramv1 r -sh`

source ${BaseDir}/SubDetScript.sh
source ${BaseDir}/UpdatePlots.sh

# Start an endless loop
while [ 1 = 1 ]
do
    cd ${BaseDir}

    # -------------------- #
    # IMPORTANT PARAMETERS #
    # -------------------- #

    AuthenticationPath="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/Authentication"
    # Database="oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE"
    Database="sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/dbfile.db"
    # Database="sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/SiStrip/dbfile.db"
    # Database="sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/SiPixel/dbfile.db"
    # Database="sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/Tracking/dbfile.db"
    # Database="sqlite_file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/RPC/dbfile.db"

    # This is the directory with the DQM root files.
    # The root files in all subdirectories will be found and processed.
    SourceDir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/data/OfflineData/Commissioning10/Cosmics/

    # The final tag name will be: TagName_DetName_TagVersion: e.g. HDQM_SiStrip_V1
    TagName="HDQM"

    # The first run from which to start populating the database
    FirstRun=122000
    # The last run for which to populate the database (-1 = all runs)
    LastRun=-1

    SubDets=(     "SiStrip" "SiPixel" "Tracking" "RPC" "L1TMonitor")
    TagVersions=( "V1"      "V1"      "V1"       "V1"  "V1" )
    #SubDets=(     "Tracking")
    #TagVersions=( "V3"      )


    # This is the directory where all the output will be copied
    StorageDir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/WWW

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

  SubDetScript ${SubDet} ${TagName}_${SubDet}_${TagVersion} ${Database} ${AuthenticationPath} ${FirstRun} ${LastRun} ${CMSSW_version} ${TemplatesDir} ${SourceDir}
	let "Update+=$?"

	let "k+=1"

	TagNames[$k]=${TagName}_${SubDet}_${TagVersion}
    done

    # Something has changed. Update all the histograms.
    if [ ${Update} -ge 1 ]; then
	# This is needed to pass the array
	argument1=`echo ${TagNames[@]}`
	argument2=`echo ${SubDets[@]}`


#  UpdatePlots "${argument1}" "${argument2}" ${CMSSW_version} ${StorageDir}


    else
	echo "Nothing changed, not redoing the plots."
    fi


    argument1=`echo ${TagNames[@]}`
    argument2=`echo ${SubDets[@]}`
    UpdatePlots "${argument1}" "${argument2}" ${CMSSW_version} ${StorageDir}


    # Wait one hour
    sleep 3600
done
