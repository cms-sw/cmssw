#! /bin/sh

export WorkDir=/home/dqmprolocal/filecopy
export YourEmail=Hyunkwan.Seo@cern.ch


#set ROOT environment
export ROOTSYS=/nfshome0/cmssw2/slc4_ia32_gcc345/lcg/root/5.18.00a-cms11/
export ROOT_DIR=${ROOTSYS}
export LD_LIBRARY_PATH=${ROOTSYS}/lib
export PATH=${ROOTSYS}/bin:${PATH}


EXE=$WorkDir/filesave_online.py
RUN_STAT=`ps -ef | grep filesave_online.py | grep -v grep | wc | awk '{print $1}'`
#DOG_STAT=`ps -ef | grep alivecheck_filesave.sh | grep -v grep | wc | awk '{print $1}'`

#if [ $DOG_STAT -gt 10 ]
#then
#    echo watchdog script seems to have some trouble at $HOST. | mail Hyunkwan.Seo@cern.ch
#    exit 0
#fi

if [ $RUN_STAT -ne 0 ]
then
    echo filesave_online.py is running at $HOST.
else
    echo filesave_online.py stopped by unknown reason and restarted now.
    LOG=$WorkDir/log/LOG.filesave.$HOST.$$
    $EXE >& $LOG &
    date >> $LOG
    echo filesave_online.py stopped by unknown reason and restarted at $HOST. >> $LOG
    echo filesave_online.py stopped by unknown reason and restarted now at $HOST. | mail $YourEmail
fi
