#!/bin/bash
#######     ------    beginning   --------  #######################
LOGFILE=/data/O2O/logs/runTestStart.log
DATE=`date`
#setting up environment variables
export HOME=/nfshome0/popconpro
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/nfshome0/popconpro/bin

echo " " | tee -a $LOGFILE
echo "----- new cronjob started for Run $1 test start at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

/bin/sh /data/O2O/scripts/RunInfoStartTest.sh $1 | tee -a $LOGFILE
/bin/sh /data/O2O/scripts/EcalDAQTest.sh | tee -a $LOGFILE
/bin/sh /data/O2O/scripts/EcalDCSTest.sh | tee -a $LOGFILE