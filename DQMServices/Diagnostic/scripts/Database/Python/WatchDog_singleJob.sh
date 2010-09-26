#!/bin/sh

CMS_PATH=/afs/cern.ch/cms
source /afs/cern.ch/cms/sw/cmsset_default.sh

# Temporary run from a test directory to validate the job using the RunRegistry to select runs.
Dir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/RunSelection/Test

# Make sure that we don't run two scripts at once (single cpu machine...)

# # StreamExpress
# # PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_StreamExpress.cfg" | awk '{print $1}'`
# PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py" | awk '{print $1}'`
# if [ ! "${PID}" ]; then
#     echo "HDQMDatabaseProducer.py for StreamExpress not running, restarting it at" `date`
#     /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_StreamExpress.cfg
# else
#     echo "HDQMDatabaseProducer.py for StreamExpress still running at" `date`
# fi
# 
# # MinimumBias
# # PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_MinimumBias.cfg" | awk '{print $1}'`
# PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py" | awk '{print $1}'`
# if [ ! "${PID}" ]; then
#     echo "HDQMDatabaseProducer.py for MinimumBias not running, restarting it at" `date`
#     /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_MinimumBias.cfg
# else
#     echo "HDQMDatabaseProducer.py for MinimumBias still running at" `date`
# fi
# 
# # Cosmics
# # PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py HDQMDatabaseProducerConfiguration_Cosmics.cfg" | awk '{print $1}'`
# PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py" | awk '{print $1}'`
# if [ ! "${PID}" ]; then
#     echo "HDQMDatabaseProducer.py for Cosmics not running, restarting it at" `date`
#     /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_Cosmics.cfg
# else
#     echo "HDQMDatabaseProducer.py for Cosmics still running at" `date`
# fi

PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py" | awk '{print $1}'`
echo "PID = ${PID}"
if [ ! "${PID}" ]; then
    echo "HDQMDatabaseProducer.py not running, starting it at" `date`
    /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_StreamExpress.cfg
    /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_MinimumBias.cfg
    /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_Cosmics.cfg
else
    echo "HDQMDatabaseProducer.py still running at" `date`
fi
