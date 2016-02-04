#!/bin/sh

# usage ./run_client.sh <run_number> <db_upload> 

eval `scramv1 runtime -sh`
cp $CMSSW_BASE/src/DQM/SiStripCommissioningDbClients/data/.OfflineDbClient.cfg client_$1.cfg
replace FILE_PATH ${SCRATCH}/source -- client_$1.cfg
replace RUN_NUMBER $1 -- client_$1.cfg
replace DB_UPDATE $2 -- client_$1.cfg 
replace CONF_DB ${CONFDB} -- client_$1.cfg 
replace DB_PARTITION ${ENV_CMS_TK_PARTITION} -- client_$1.cfg 
cmsRun client_$1.cfg
mkdir ${SCRATCH}/client/$1
mv ${SCRATCH}/*$1*.root ${SCRATCH}/client/$1
mv client_$1.cfg ${SCRATCH}/client/$1
mv event.log ${SCRATCH}/client/$1/event_$1.log
mv error.log ${SCRATCH}/client/$1/error_$1.log
mv debug.log ${SCRATCH}/client/$1/debug_$1.log
