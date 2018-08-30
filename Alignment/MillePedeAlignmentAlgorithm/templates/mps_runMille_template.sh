#!/bin/zsh 
#
# Run script template for Mille jobs
#
# Adjustments might be needed for CMSSW environment.
#
# In the very beginning of this script, stager requests for the files will be added.

# these defaults will be overwritten by MPS
RUNDIR=$HOME/scratch0/some/path
MSSDIR=/castor/cern.ch/user/u/username/another/path
MSSDIRPOOL=

clean_up () {
#try to recover log files and root files
    echo try to recover log files and root files ...
    cp -p *.log $RUNDIR
    cp -p *.log.gz $RUNDIR
    cp -p millePedeMonitor*root $RUNDIR
    exit
}
#LSF signals according to http://batch.web.cern.ch/batch/lsf-return-codes.html
trap clean_up HUP INT TERM SEGV USR2 XCPU XFSZ IO

# a helper function to repeatedly try failing copy commands
untilSuccess () {
# trying "${1} ${2} ${3} > /dev/null" until success, if ${4} is a
# positive number run {1} with -f flag,
# break after ${5} tries (with four arguments do up to 5 tries).
    if  [[ ${#} -lt 4 || ${#} -gt 5 ]]
    then
        echo ${0} needs 4 or 5 arguments
        return 1
    fi

    TRIES=0
    MAX_TRIES=5
    if [[ ${#} -eq 5 ]]
    then
        MAX_TRIES=${5}
    fi


    if [[ ${4} -gt 0 ]]
    then 
        ${1} -f ${2} ${3} > /dev/null
    else 
        ${1} ${2} ${3} > /dev/null
    fi
    while [[ ${?} -ne 0 ]]
    do # if not successfull, retry...
        if [[ ${TRIES} -ge ${MAX_TRIES} ]]
        then # ... but not until infinity!
            if [[ ${4} -gt 0 ]]
            then
                echo ${0}: Give up doing \"${1} -f ${2} ${3} \> /dev/null\".
                return 1
            else
                echo ${0}: Give up doing \"${1} ${2} ${3} \> /dev/null\".
                return 1
            fi
        fi
        TRIES=$((${TRIES}+1))
        if [[ ${4} -gt 0 ]]
        then
            echo ${0}: WARNING, problems with \"${1} -f ${2} ${3} \> /dev/null\", try again.
            sleep $((${TRIES}*5)) # for before each wait a litte longer...
            ${1} -f ${2} ${3} > /dev/null
        else
            echo ${0}: WARNING, problems with \"${1} ${2} ${3} \> /dev/null\", try again.
            sleep $((${TRIES}*5)) # for before each wait a litte longer...
            ${1} ${2} ${3} > /dev/null
        fi
    done

    if [[ ${4} -gt 0 ]]
    then
        echo successfully executed \"${1} -f ${2} ${3} \> /dev/null\"
    else
        echo successfully executed \"${1} ${2} ${3} \> /dev/null\"
    fi
    return 0
}

export X509_USER_PROXY=${RUNDIR}/.user_proxy


# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo "Running at $(date) \n        on $HOST \n        in directory $BATCH_DIR."

# set up the CMS environment (choose your release and working area):
cd CMSSW_RELEASE_AREA
echo Setting up $(pwd) as CMSSW environment. 
eval `scram runtime -sh`
rehash

cd $BATCH_DIR
echo The running directory is $(pwd).
# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

gzip -f *.log
gzip milleBinaryISN.dat
echo "\nDirectory content after running cmsRun and zipping log+dat files:"
ls -lh 
# Copy everything you need to MPS directory of your job,
# but you might want to copy less stuff to save disk space
# (separate cp's for each item, otherwise you loose all if one file is missing):
cp -p *.log.gz $RUNDIR
# store  millePedeMonitor also in $RUNDIR, below is backup in $MSSDIR
cp -p millePedeMonitor*root $RUNDIR

# Copy MillePede binary file to Castor
# Must use different command for the cmscafuser pool
if [ "$MSSDIRPOOL" != "cmscafuser" ]; then
# Not using cmscafuser pool => rfcp command must be used
  export STAGE_SVCCLASS=$MSSDIRPOOL
  export STAGER_TRACE=
  nsrm -f $MSSDIR/milleBinaryISN.dat.gz
  echo "rfcp milleBinaryISN.dat.gz $MSSDIR/"
  untilSuccess rfcp milleBinaryISN.dat.gz   $MSSDIR/ 0
  untilSuccess rfcp treeFile*root         $MSSDIR/treeFileISN.root 0
  untilSuccess rfcp millePedeMonitor*root $MSSDIR/millePedeMonitorISN.root 0
else
  MSSCAFDIR=`echo $MSSDIR | perl -pe 's/\/castor\/cern.ch\/cms//gi'`
  # ensure the directories exists
  mkdir -p ${MSSCAFDIR}/binaries
  mkdir -p ${MSSCAFDIR}/tree_files
  mkdir -p ${MSSCAFDIR}/monitors
  # copy the files
  echo "xrdcp -f milleBinaryISN.dat.gz ${MSSCAFDIR}/binaries/milleBinaryISN.dat.gz > /dev/null"
  untilSuccess xrdcp milleBinaryISN.dat.gz    ${MSSCAFDIR}/binaries/milleBinaryISN.dat.gz  1
  untilSuccess xrdcp treeFile.root            ${MSSCAFDIR}/tree_files/treeFileISN.root 1
  untilSuccess xrdcp millePedeMonitorISN.root ${MSSCAFDIR}/monitors/millePedeMonitorISN.root 1
fi
