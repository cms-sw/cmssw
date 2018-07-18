#!/bin/bash
#######     ------    beginning   --------  #######################
LOGFILE=@root/logs/runTestStart.log
DATE=`date`
#setting up environment variables
export HOME=@home
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:@home/bin

echo " " | tee -a $LOGFILE
echo "----- new cronjob started for Run $1 test start at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

/bin/sh @root/scripts/RunInfoStartTest.sh $1 | tee -a $LOGFILE
/bin/sh @root/scripts/EcalDAQTest.sh | tee -a $LOGFILE
/bin/sh @root/scripts/EcalDCSTest.sh | tee -a $LOGFILE