#! /bin/sh

WorkDir=/data/dqm/filereg
CMSSW_V=CMSSW_2_2_10
YourEmail=lilopera@cern.ch


#export CVS_RSH=ssh
#export CVSROOT=:ext:cmscvs.cern.ch:/cvs_server/repositories/CMSSW
source ~cmssw2/cmsset_default.sh
cd $WorkDir/$CMSSW_V/src
cmsenv
XPYTHONPATH=$PYTHONPATH
source /home/dqm/rpms/slc4_ia32_gcc345/cms/dqmgui/4.6.0/etc/profile.d/env.sh
export PYTHONPATH=$XPYTHONPATH:$PYTHONPATH

#export TNS_ADMIN=/nfshome0/xiezhen/conddb


EXE=$WorkDir/dqmPostProcessing_online.py
RUN_STAT=`ps -ef | grep dqmPostProcessing_online.py | grep -v grep | wc | awk '{print $1}'`
#DOG_STAT=`ps -ef | grep alivecheck_dqmPostProcessing.sh | grep -v grep | wc | awk '{print $1}'`

#if [ $DOG_STAT -gt 10 ]
#    then echo watchdog for dqmPostProcessing seems to have some trouble at $HOST. | mail $YourEmail
#    exit 0
#fi


if [ $RUN_STAT -ne 0 ]
then
    echo dqmPostProcessing_online.py is running
else
    echo dqmPostProcessing_online.py stopped by unknown reason and restarted now.
    LOG=$WorkDir/log/LOG.postprocess.$$
    $EXE >& $LOG &
    date >> $LOG
    echo dqmPostProcessing_online.py stopped by unknown reason and restarted at $HOST. >> $LOG
    echo dqmPostProcessing_online.py stopped by unknown reason and restarted now at $HOST. | mail $YourEmail
fi

