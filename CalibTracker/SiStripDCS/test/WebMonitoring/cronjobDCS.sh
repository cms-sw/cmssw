#!/bin/bash

if [ -f /afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/cronlog/LockFileDCSTmp ]; then
    exit
fi

LOCKFILE=/afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/cronlog/LockFileDCSTmp
ERRORLOCKFILE="/afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/cronlog/ErrorLockFileDCS_`date +%Y%m%d%H%M`"

touch $LOCKFILE
touch $ERRORLOCKFILE

trap "rm -f $LOCKFILE" EXIT

export PATH=$PATH:/afs/cern.ch/cms/sw/common/
export CMS_PATH=/afs/cern.ch/cms
export FRONTIER_PROXY=http://cmst0frontier.cern.ch:3128
#export FRONTIER_FORCERELOAD=long # This should not be used anymore!!!
export SCRAM_ARCH=slc5_ia32_gcc434

cd /afs/cern.ch/cms/tracker/sistrcalib/DCSTrend/CMSSW_3_11_0/src
eval `scramv1 runtime -sh`

echo "My Scram Variable"
echo $SCRAM_ARCH

cd /afs/cern.ch/cms/tracker/sistrcalib/DCSTrend
./RunCheckAllIOVs.sh
./UpdateRuns.sh

rm -f $LOCKFILE
rm -f $ERRORLOCKFILE
