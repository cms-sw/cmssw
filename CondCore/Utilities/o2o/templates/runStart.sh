#!/bin/bash
#######     ------    beginning   --------  #######################
LOGFILE=@root/logs/runStart.log
DATE=`date`
#setting up environment variables
export HOME=@home
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:@home/bin

echo " " | tee -a $LOGFILE
echo "----- new job started for run $1 start at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

/bin/sh @root/scripts/RunInfoStart.sh $1 | tee -a $LOGFILE
/bin/sh @root/scripts/EcalDAQ.sh | tee -a $LOGFILE
/bin/sh @root/scripts/EcalDCS.sh | tee -a $LOGFILE

