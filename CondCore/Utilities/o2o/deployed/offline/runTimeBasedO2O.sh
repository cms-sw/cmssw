#!/bin/bash
#######     ------    beginning   --------  #######################
BASEDIR=/data/O2O
LOGFILE=${BASEDIR}/logs/TimeBasedO2O.log
DATE=`date`
echo " " | tee -a $LOGFILE
echo "----- new cronjob started for Time Based O2O at -----" | tee -a $LOGFILE
echo "$DATE" | tee -a $LOGFILE

#Check if the exportation scripts are running, if so exits without launching them
PID_0=`ps aux | grep '/bin/sh /data/O2O/scripts/EcalLaser.sh' | grep -v grep | awk '{print $2}'`
if [ "${PID_0}" ]
    then
    echo "Ecal Laser Exportation script still running with pid ${PID_0}: exiting" | tee -a $LOGFILE
    exit 1
else
    /bin/sh /data/O2O/scripts/EcalLaser.sh
fi
PID_1=`ps aux | grep '/bin/sh /data/O2O/scripts/EcalLaser_express.sh' | grep -v grep | awk '{print $2}'`
if [ "${PID_1}" ]
    then
    echo "Ecal LaserExpress Exportation script still running with pid ${PID_1}: exiting" | tee -a $LOGFILE
    exit 1
else
    /bin/sh /data/O2O/scripts/EcalLaser_express.sh
fi
