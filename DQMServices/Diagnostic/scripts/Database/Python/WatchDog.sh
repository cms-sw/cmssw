#!/bin/sh

CMS_PATH=/afs/cern.ch/cms
source /afs/cern.ch/cms/sw/cmsset_default.sh

# Temporary run from a test directory to validate the job using the RunRegistry to select runs.
Dir=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/HDQM/Cron/Scripts/RunSelection/Test

# Make sure that we don't run two scripts at once (single cpu machine...)

# StreamExpress
PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_StreamExpress.cfg" | awk '{print $1}'`
if [ ! "${PID}" ]; then
    echo "HDQMDatabaseProducer.py for StreamExpress not running, restarting it at" `date`
    /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_StreamExpress.cfg
else
    echo "HDQMDatabaseProducer.py for StreamExpress still running at" `date`
fi

# MinimumBias
PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_MinimumBias.cfg" | awk '{print $1}'`
if [ ! "${PID}" ]; then
    echo "HDQMDatabaseProducer.py for MinimumBias not running, restarting it at" `date`
    /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_MinimumBias.cfg
else
    echo "HDQMDatabaseProducer.py for MinimumBias still running at" `date`
fi

# Cosmics
PID=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_Cosmics.cfg" | awk '{print $1}'`
if [ ! "${PID}" ]; then
    echo "HDQMDatabaseProducer.py for Cosmics not running, restarting it at" `date`
    /usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_Cosmics.cfg
else
    echo "HDQMDatabaseProducer.py for Cosmics still running at" `date`
fi

# RPC-Full production

# Nothing else must be running
PID1=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_StreamExpress.cfg" | awk '{print $1}'`
PID2=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_MinimumBias.cfg" | awk '{print $1}'`
PID3=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_Cosmics.cfg" | awk '{print $1}'`
# StreamExpress
if [ ! "${PID1}" ]; then
    if [ ! "${PID2}" ]; then
	if [ ! "${PID3}" ]; then
	    PIDRPC1=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_StreamExpressRPC.cfg" | awk '{print $1}'`
	    PIDRPC2=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_MinimumBiasRPC.cfg" | awk '{print $1}'`
	    PIDRPC3=`ps ax | grep -v grep | grep "${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_CosmicsRPC.cfg" | awk '{print $1}'`
	    if [ ! "${PIDRPC1}" ]; then
		if [ ! "${PIDRPC2}" ]; then
		    if [ ! "${PIDRPC3}" ]; then

			echo "HDQMDatabaseProducer.py for StreamExpress not running, restarting it at" `date`
			/usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_StreamExpressRPC.cfg
			/usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_MinimumBiasRPC.cfg
			/usr/bin/python ${Dir}/HDQMDatabaseProducer.py ${Dir}/HDQMDatabaseProducerConfiguration_CosmicsRPC.cfg
		    fi
		fi
	    fi
	fi
    fi
fi