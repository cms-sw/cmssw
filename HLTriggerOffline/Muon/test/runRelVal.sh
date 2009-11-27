#!/bin/bash

# Accepts a DBS path, so use looks like:
# runRelVal.sh /RelValZMM/CMSSW_3_1_1-STARTUP31X_V1-v2/GEN-SIM-RECO

# This script creates a copy of hltMuonValidator_cfg.py and includes the
# root files from the chosen sample in the PoolSource, then runs the new 
# ana.py file.  It then runs the copies hltMuonPostProcessor_cfg.py into 
# post.py using the ouput from the analyzer job.

# Use the --post (or -p) option to only run the post processing, using
# validation content already existing in the DBS dataset.

# Use the -n option to specify how many events to run over

getFiles () {
    FILE_NAMES=`mktemp`
    $DBS_CMD lsf --path=$1 | grep .root | \
        sed "s:\(/store/.*\.root\):'\1', :" >> $FILE_NAMES
    echo $FILE_NAMES
}

POST_ONLY=false
N_EVENTS=-1

LONGOPTSTRING=`getopt --long post -o pn: -- "$@"`
eval set -- "$LONGOPTSTRING"
while true ; do
    case "$1" in
        --post) POST_ONLY=true ; shift ;;
        -p)     POST_ONLY=true ; shift ;;
        -n)     shift ; N_EVENTS=$1 ; shift ;;
        --)     shift ; break ;;
        *)      echo "Internal error!" ; exit 1 ;;
    esac
done



if [ -z "$CMSSW_VERSION" ] ; then 
   echo "CMSSW environment not set up; run cmsenv"
   exit
else
   DBS_CMD="python $DBSCMD_HOME/dbsCommandLine.py -c "
fi

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
if [[ ! -z "$HLTDEBUGPATH" ]] ; then echo $HLTDEBUGPATH; fi
if [[ ! -z "$RECOPATH"     ]] ; then echo $RECOPATH    ; fi

RECOFILES="$(getFiles $RECOPATH)"
HLTDEBUGFILES="$(getFiles $HLTDEBUGPATH)"

if [ $POST_ONLY = true ]; then
    cp hltMuonPostProcessor_cfg.py post.py
    echo "" >> post.py
    echo "process.source.fileNames = [" >> post.py
    cat $RECOFILES >> post.py
    echo "]" >> post.py
    echo "" >> post.py
    echo "process.maxEvents.input = $N_EVENTS" >> post.py
    cmsRun post.py
    LONGNAME=$RECOPATH
else
    cp hltMuonValidator_cfg.py ana.py
    echo "" >> ana.py
    echo "process.source.fileNames = [" >> ana.py
    cat $RECOFILES >> ana.py
    echo "]" >> ana.py
    echo "process.source.secondaryFileNames = [" >> ana.py
    cat $HLTDEBUGFILES >> ana.py
    echo "]" >> ana.py
    echo "process.maxEvents.input = $N_EVENTS" >> ana.py
    cmsRun ana.py
    cmsRun hltMuonPostProcessor_cfg.py
    LONGNAME=$RECOPATH
fi

SHORTNAME=`echo $LONGNAME | sed "s/\/RelVal\(.*\)\/CMSSW_\(.*\)\/.*/\1_\2/"`
mv hltMuonPostProcessor.root $SHORTNAME.root
echo "Produced $SHORTNAME.root"
