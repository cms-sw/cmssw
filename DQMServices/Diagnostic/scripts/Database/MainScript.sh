#!/bin/sh

# Define the release version to use
CMSSW_version=/home/cmstacuser/historyDQM/CMSSW_Releases/CMSSW_3_2_2

cd ${CMSSW_version}/src
eval `scramv1 r -sh`
cd -

# -------------------- #
# IMPORTANT PARAMETERS #
# -------------------- #

AuthenticationPath="/home/cmstacuser/historyDQM/Cron/Scripts/Authentication"
Database="oracle://cms_orcoff_prep/CMS_COND_STRIP"
# Database="oracle://cms_orcoff_prep/CMS_DQM_31X_OFFLINE"
# Database="sqlite_file:dbfile.db"

# The final tag name will be: TagName_DetName_TagVersion: e.g. HDQM_SiStrip_V1
TagName="HDQM"

# The first run from which to start populating the database
FirstRun=108239
# The last run for which to populate the database (-1 = all runs)
LastRun=-1

SubDets=(     "SiStrip" "SiPixel" "Tracking")
TagVersions=( "V2"      "V2"      "V1"      )

# -------------------- #
# -------------------- #

k=0
SubDetNumber=${#SubDets[*]}

while [ "$k" -lt "$SubDetNumber" ]
    do
    SubDet=${SubDets[k]}
    TagVersion=${TagVersions[k]}
    echo "Processing new runs for $SubDet"
    ./SubDetScript.sh $SubDet ${TagName}_${SubDet}_${TagVersion} ${Database} ${AuthenticationPath} ${FirstRun} ${LastRun}

    let "k+=1"
done
