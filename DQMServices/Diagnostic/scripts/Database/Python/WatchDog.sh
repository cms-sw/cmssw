#!/bin/sh

# Temporary run from a test directory to validate the job using the RunRegistry to select runs.
Dir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/RunSelection/Test

# Check if process is running, if not launch it again
PID=`/sbin/pidof -x WatchDog.sh`
if [ "${PID}" ]; then
    echo "Still running from previous call at" `date`
    exit
fi

# StreamExpress
PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_StreamExpress.cfg" | awk '{print $1}'`
if [ ! "${PID}" ]; then
    echo "HDQMDatabaseProducer.py for StreamExpress not running, restarting it at" `date`
    python ${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_StreamExpress.cfg
else
    echo "HDQMDatabaseProducer.py for StreamExpress still running at" `date`
fi

# MinimumBias
PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_MinimumBias.cfg" | awk '{print $1}'`
if [ ! "${PID}" ]; then
    echo "HDQMDatabaseProducer.py for MinimumBias not running, restarting it at" `date`
    python ${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_MinimumBias.cfg
else
    echo "HDQMDatabaseProducer.py for MinimumBias still running at" `date`
fi

# Cosmics
PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_Cosmics.cfg" | awk '{print $1}'`
if [ ! "${PID}" ]; then
    echo "HDQMDatabaseProducer.py for MinimumBias not running, restarting it at" `date`
    python ${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_Cosmics.cfg
else
    echo "HDQMDatabaseProducer.py for MinimumBias still running at" `date`
fi
