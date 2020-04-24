#! /bin/sh

WorkDir=/data/dqm/filereg
YourEmail=lilopera@cern.ch
source /nfshome0/cmssw2/scripts/setup.sh
source /home/dqm/rpms/slc4_ia32_gcc345/cms/dqmgui/4.6.0/etc/profile.d/env.sh


EXE=$WorkDir/fileTransfer.py
RUN_STAT=`ps -ef | grep fileTransfer.py | grep -v grep | wc | awk '{print $1}'`
#DOG_STAT=`ps -ef | grep alivecheck_dqmPostProcessing.sh | grep -v grep | wc | awk '{print $1}'`

#if [ $DOG_STAT -gt 10 ]
#    then echo watchdog for dqmPostProcessing seems to have some trouble at $HOST. | mail $YourEmail
#    exit 0
#fi


if [ $RUN_STAT -ne 0 ]
then
    echo fileTransfer.py is running
else
    echo fileTransfer.py stopped by unknown reason and restarted now.
    LOG=$WorkDir/log/LOG.fileTransfer.$$
    $EXE >& $LOG &
    date >> $LOG
    echo dfileTransfer.py stopped by unknown reason and restarted at $HOSTNAME. >> $LOG
    echo fileTransfer.py stopped by unknown reason and restarted now at $HOSTNMAE. | mail $YourEmail
fi
