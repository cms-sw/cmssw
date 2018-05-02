#!/bin/bash
#######     ------    beginning   --------  #######################
LOGFILE=@root/logs/runTestStop.log
DATE=`date`
#setting up environment variables
export HOME=@home
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:@home/bin

echo " " | tee -a $LOGFILE
echo "----- new job started for run $1 test stop at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

/bin/sh @root/scripts/RunInfoStopTest.sh $1 | tee -a $LOGFILE
