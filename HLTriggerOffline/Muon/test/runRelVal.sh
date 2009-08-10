#!/bin/bash

# Accepts a DBS path, so use looks like:
# runRelVal.sh /RelValZMM/CMSSW_3_1_1-STARTUP31X_V1-v2/GEN-SIM-RECO

# You must have the DBS CLI (command line interface) set up to use this script.
# To set it up on lxplus, run:
# source /afs/cern.ch/user/v/valya/scratch0/setup.sh

# This script creates a copy of muonTriggerRateTimeAnalyzer_cfg.py and includes
# the root files from the chosen sample in the PoolSource, then runs the new
# ana.py file.  It then runs the copies PostProcessor_cfg.py into post.py 
# using the ouput from the analyzer job.

# Use the --post (or -p) option to only run the post processing, using
# validation content already existing in the DBS dataset.

POST_ONLY=false

LONGOPTSTRING=`getopt --long post -o p -- "$@"`
eval set -- "$LONGOPTSTRING"
while true ; do
    case "$1" in
        --post) POST_ONLY=true ; shift ;;
        -p)     POST_ONLY=true ; shift ;;
        --)     shift ; break ;;
        *)      echo "Internal error!" ; exit 1 ;;
    esac
done

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

echo "Using dataset(s): "
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

if [ $POST_ONLY = true ]; then
    cat PostProcessor_cfg.py | \
	sed "s:\(fileNames.*\)vstring(.*):\1vstring($FILES):" > post.py
    cmsRun post.py
    LONGNAME=$RECOPATH
else
    cat muonTriggerRateTimeAnalyzer_cfg.py | \
      sed "s:\(fileNames.*\)vstring():\1vstring($FILES):" | \
      sed "s:\(secondaryFileNames.*\)vstring():\1vstring($SECFILES):" > ana.py
    cmsRun ana.py
    cmsRun PostProcessor_cfg.py
    LONGNAME=$HLTDEBUGPATH
fi

SHORTNAME=`echo $LONGNAME | sed "s/\/RelVal\(.*\)\/CMSSW_\(.*\)\/.*/\1_\2/"`
mv PostProcessor.root $SHORTNAME.root
