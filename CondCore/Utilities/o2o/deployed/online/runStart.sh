#!/bin/bash
#######     ------    beginning   --------  #######################
LOGFILE=/data/O2O/logs/runStart.log
DATE=`date`
#setting up environment variables
export HOME=/nfshome0/popconpro
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/nfshome0/popconpro/bin

echo " " | tee -a $LOGFILE
echo "----- new job started for run $1 start at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

/bin/sh /data/O2O/scripts/RunInfoStart.sh $1 | tee -a $LOGFILE
/bin/sh /data/O2O/scripts/EcalDAQ.sh | tee -a $LOGFILE
/bin/sh /data/O2O/scripts/EcalDCS.sh | tee -a $LOGFILE
#/bin/sh /data/O2O/scripts/RunInfoStartTest.sh $1 | tee -a /data/O2O/logs/runTestStart.log
