#!/bin/bash

# Accepts a DBS path, so use looks like:
# runRelVal.sh /RelValZMM/CMSSW_2_1_10_STARTUP_V7_v2/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO

# This script creates a copy of muonTriggerRateTimeAnalyzer_cfg.py and includes
# the root files from the chosen sample in the PoolSource, then runs the new
# ana.py file.

# You must have the DBS CLI (command line interface) set up to use this script.
# To set it up on lxplus, run:
# source /afs/cern.ch/user/v/valya/scratch0/setup.sh

if [[ $1 =~ .*GEN-SIM-RECO ]]; then
    HLTDEBUGPATH=`echo $1 | sed 's/GEN-SIM-RECO/GEN-SIM-DIGI-RAW-HLTDEBUG/'`
    RECOPATH=$1
elif [[ $1 =~ .*HLTDEBUG ]]; then
    HLTDEBUGPATH=$1
    RECOPATH=`echo $1 | sed 's/GEN-SIM-DIGI-RAW-HLTDEBUG/GEN-SIM-RECO/'`
elif [[ $1 =~ .*FastSim.* ]]; then
    HLTDEBUGPATH=
    RECOPATH=$1
else
    echo "The given path does not appear to be valid.  Exiting."
    exit
fi

echo "Using datasets: "
echo $HLTDEBUGPATH
echo $RECOPATH


if [ "$DBSCMD_HOME" ] ; then 
    DBS_CMD="python $DBSCMD_HOME/dbsCommandLine.py -c " 
elif [ "$DBS_CLIENT_ROOT" ] ; then
    DBS_CMD="python $DBS_CLIENT_ROOT/lib/DBSAPI/dbsCommandLine.py -c" 
else 
    echo "Cannot setup DBS command line interface. Exiting."
    exit
fi

FILES=`$DBS_CMD lsf --path=$RECOPATH | grep .root | sed "s:\(/store/.*\.root\):'\1',\n:"`
FILES=$FILES,,,
FILES=`echo $FILES | sed 's/,,,,//'`

if [[ ! -z "$HLTDEBUGPATH" ]] ; then 
    SECFILES=`$DBS_CMD lsf --path=$HLTDEBUGPATH | grep .root | sed "s:\(/store/.*\.root\):'\1',\n:"`
    SECFILES=$SECFILES,,,
    SECFILES=`echo $SECFILES | sed 's/,,,,//'`
else
    SECFILES=
fi

cat muonTriggerRateTimeAnalyzer_cfg.py | \
    sed "s:\(fileNames.*\)vstring():\1vstring($FILES):" | \
    sed "s:\(secondaryFileNames.*\)vstring():\1vstring($SECFILES):" > ana.py

cmsRun ana.py
cmsRun PostProcessor_cfg.py

jobName=`echo $HLTDEBUGPATH | sed "s/\/RelVal\(.*\)\/CMSSW_\(.*\)\/.*/\1_\2/"`
mv PostProcessor.root $jobName.root
