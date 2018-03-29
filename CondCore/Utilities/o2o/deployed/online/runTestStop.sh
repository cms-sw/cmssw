#!/bin/bash
#######     ------    beginning   --------  #######################
LOGFILE=/data/O2O/logs/runTestStop.log
DATE=`date`
#setting up environment variables
export HOME=/nfshome0/popconpro
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/nfshome0/popconpro/bin

echo " " | tee -a $LOGFILE
echo "----- new job started for run $1 test stop at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

/bin/sh /data/O2O/scripts/RunInfoStopTest.sh $1 | tee -a $LOGFILE
