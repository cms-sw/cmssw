#!/bin/sh

# This scripts checks if the MainScript.sh is running and if not it launches it again.
#count=`ps aux | grep -c MainScript`
#
#if [ $count -ge 2 ]; then
#  echo "Still running at" `date`
#else
#  echo "Not running, restarting it at" `date`
#  /home/cmstacuser/historyDQM/Cron/Scripts/MainScript.sh > log &
#fi

Dir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts

# Check if process is running, if not launch it again
PID=`/sbin/pidof -x MainScript.sh`
if [ ! "${PID}" ]; then
    echo "Not running, restarting it at" `date`
    ${Dir}/MainScript.sh &
else
    echo "Still running at" `date`
fi
