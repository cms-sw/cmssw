#!/bin/bash

if [ -f /afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/cronlog/LockFileTmp ]; then
    exit
fi

LOCKFILE=/afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/cronlog/LockFileTmp
ERRORLOCKFILE="/afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/cronlog/ErrorLockFile_`date +%Y%m%d%H%M`"

touch $LOCKFILE
touch $ERRORLOCKFILE

trap "rm -f $LOCKFILE" EXIT

export PATH=$PATH:/afs/cern.ch/cms/sw/common/
export CMS_PATH=/afs/cern.ch/cms
#export FRONTIER_FORCERELOAD=long # This should not be used anymore!!!

cd /afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/CMSSW_3_2_5/src/
eval `scramv1 runtime -sh`

cd /afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB
./MonitorDB_NewDirStructure.sh cms_orcoff_prod CMS_COND_31X_STRIP CMS_COND_31X_GLOBALTAG FrontierProd
#./MonitorDB_NewDirStructure.sh cms_orcoff_prod CMS_COND_31X_FROM21X CMS_COND_31X_GLOBALTAG FrontierProd
./MonitorDB_NewDirStructure.sh cms_orcoff_prep CMS_COND_STRIP CMS_COND_30X_GLOBALTAG FrontierPrep
./MonitorDB_NewDirStructure.sh cms_orcoff_prep CMS_COND_31X_ALL CMS_COND_30X_GLOBALTAG FrontierPrep
./MonitorDB_NewDirStructure.sh cms_orcoff_prep CMS_COND_30X_STRIP CMS_COND_30X_GLOBALTAG FrontierPrep
./Monitor_NoiseRatios.sh cms_orcoff_prod CMS_COND_31X_STRIP CMS_COND_31X_GLOBALTAG FrontierProd
./Monitor_RunInfo.sh cms_orcoff_prod CMS_COND_31X_STRIP CMS_COND_31X_GLOBALTAG CMS_COND_31X_RUN_INFO FrontierProd

#Not needed anymore, since the 21X tags won't change (they are not in use anymore)
#cd /afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB/CMSSW_2_2_6/src/
#eval `scramv1 runtime -sh`

#cd /afs/cern.ch/cms/tracker/sistrcalib/MonitorConditionDB
#./MonitorDB_NewDirStructure.sh cms_orcoff_prod CMS_COND_21X_STRIP CMS_COND_21X_GLOBALTAG FrontierProd

rm -f $LOCKFILE
rm -f $ERRORLOCKFILE
