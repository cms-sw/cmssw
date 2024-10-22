#!/bin/sh

# deployment-specific params
BASEDIR=/data/O2O
RELEASE=CMSSW_10_0_5
RELEASEDIR=/cvmfs/cms.cern.ch/slc6_amd64_gcc630/cms/cmssw/${RELEASE}

# command params
OPTIND=1

SUBSYS=""
JOBNAME=""

while getopts "h?s:j:" opt; do
    case $opt in
    h|\?)
	echo "Mo' to spiego..."
        exit 0
        ;;
    s)  SUBSYS=$OPTARG
	;;
    j)  JOBNAME=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

O2ONAME=$SUBSYS$JOBNAME
#echo "name=$O2ONAME, subsystem=$SUBSYS, job=$JOBNAME"

# o2o specific params
LOGFILE=${BASEDIR}/logs/$O2ONAME.log
JOBDIR=${BASEDIR}/${SUBSYS}/${JOBNAME}
DATE=`date`

# functions
function logRun(){
    echo "----- new job started for $1 at -----" | tee -a $LOGFILE
    echo $DATE | tee -a $LOGFILE
}

function log() {
    echo "[`date`] : $@ " | tee -a $LOGFILE
}

function submit_command() {
    logRun $1
    o2o run -n $1 "$2" | tee -a $LOGFILE
}

function submit_test_command() {
    logRun $1
    o2o --db dev run -n $1 "$2" | tee -a $LOGFILE
}

function submit_cmsRun() {
    COMMAND="cmsRun $2 destinationDatabase={db} destinationTag={tag}"
    logRun $1
    o2o run -n $1 "$COMMAND" | tee -a $LOGFILE
}

function submit_test_cmsRun() {
    COMMAND="cmsRun $2 destinationDatabase={db} destinationTag={tag}"
    logRun $1
    o2o --db dev run -n $1 "$COMMAND" | tee -a $LOGFILE
}

function submit_popCon() {
    COMMAND="popconRun $2 -d {db} -t {tag} -c"
    logRun $1
    o2o run -n $1 "$COMMAND"  | tee -a $LOGFILE
}

function submit_test_popCon() {
    COMMAND="popconRun $2 -d {db} -t {tag} -c"
    logRun $1
    o2o --db dev run -n $1 "$COMMAND"  | tee -a $LOGFILE
}

# global variables
export PYTHON_EGG_CACHE=/data/condbpro
export SCRAM_ARCH=slc6_amd64_gcc630
export O2O_LOG_FOLDER=/data/O2O/logs/${O2ONAME}
export COND_AUTH_PATH=$BASEDIR
source /cvmfs/cms.cern.ch/cmsset_default.sh

cd ${RELEASEDIR}/src
eval `scramv1  run -sh`
# set up OCCI workaround
export LD_PRELOAD=$CMS_ORACLEOCCI_LIB

# workaround for oracle tnsnames                                                                                                                   
export TNS_ADMIN=/cvmfs/cms.cern.ch/slc6_amd64_gcc530/cms/oracle-env/29/etc

cd ${JOBDIR}


