#!/bin/sh

# deployment-specific params
BASEDIR=@root
RELEASE=@release
ARCH=@arch
RELEASEDIR=@cmsswroot/${RELEASE}

# command params
OPTIND=1

JOBNAME=""

while getopts "h?s:j:" opt; do
    case $opt in
    h|\?)
	echo "Mo' to spiego..."
        exit 0
        ;;
    j)  JOBNAME=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

# o2o specific params
LOGFILE=${BASEDIR}/logs/$JOBNAME.log
JOBDIR=${BASEDIR}/${JOBNAME}
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
export PYTHON_EGG_CACHE=@localhome
export SCRAM_ARCH=$ARCH
export O2O_LOG_FOLDER=@root/logs/${JOBNAME}
export COND_AUTH_PATH=$BASEDIR
source @cmsswroot/cmsset_default.sh

cd ${RELEASEDIR}/src
eval `scramv1  run -sh`
# set up OCCI workaround
export LD_PRELOAD=$CMS_ORACLEOCCI_LIB

# workaround for oracle tnsnames
export TNS_ADMIN=@extroot/oracle-env/29/etc

cd ${JOBDIR}


