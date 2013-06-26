#!/bin/bash

# initialize the base number of jobs
let baseJobCount=0

# initialize the forced restart time
let forcedRestartTime=`date '+%s'`
let restartTimeout=604800  # one week (needs to be less than range in logfile cleanup script)

# initialize the logfile cleanup time
let logfileCleanupTime=$forcedRestartTime
let cleanupTimeout=86400  # one day

# start an initial monitor consumer
TIMESTAMP=`date '+%Y%m%d%H%M%S'`
CONSUMER_LOGFILE="../log/client2/CONS${TIMESTAMP}.log"
(cd ../log/client2 ; cmsRun ../../cfg/eventConsumer.py) >& $CONSUMER_LOGFILE &
sleep 10

while [ 1 -eq 1 ]
do
    # fetch the current time (seconds since 1970)
    let currentTime=`date '+%s'`

    # touch the shared memory key file to keep it current
    touch $SMPB_SHM_KEYFILE

    # count the number of background jobs we have running
    rm -f jobList.tmp >& /dev/null
    jobs >& jobList.tmp
    let jobCount=`cat jobList.tmp | wc -l`
    #echo "jobCount = $jobCount"

    # check that the monitor consumer is still receiving events
    let activeConsumerCount=`find ../log/client2 -name '*.log' -mmin -3 -print | wc -l`
    #echo "active consumer count = $activeConsumerCount"

    # if the background job has died or if it is time to
    # force a restart, restart everything
    if [ $(($jobCount - $baseJobCount)) -lt 1 ] || \
       [ $activeConsumerCount -lt 1 ] || \
       [ $(($currentTime - $forcedRestartTime)) -gt $restartTimeout ]
    then
        let forcedRestartTime=$currentTime
        mv ../log/watchdog/alive ../log/watchdog/alive.old
        echo `date`
        echo "A restart is needed, working on that."
        echo `date` > ../log/watchdog/alive
        echo "A restart is needed, working on that." >> ../log/watchdog/alive

        # restart the BU/FU/SM
        ./restartPlayback.sh

        # count the baseline number of background jobs
        rm -f jobList.tmp >& /dev/null
        jobs >& jobList.tmp
        let baseJobCount=`cat jobList.tmp | wc -l`
        #echo "base = $baseJobCount"

        # start a fresh monitor consumer
        TIMESTAMP=`date '+%Y%m%d%H%M%S'`
        CONSUMER_LOGFILE="../log/client2/CONS${TIMESTAMP}.log"
        (cd ../log/client2 ; cmsRun ../../cfg/eventConsumer.py) >& $CONSUMER_LOGFILE &

        # sleep for an extra time (to prevent rapid restarts in case of bugs)
        let extraLoopCount=10
        while [ $extraLoopCount -gt 0 ]
        do
            mv ../log/watchdog/alive ../log/watchdog/alive.old
            echo `date`
            echo "Restart attempted, will return to the normal checkup loop in $extraLoopCount minutes."
            echo `date` > ../log/watchdog/alive
            echo "Restart attempted, will return to the normal checkup loop in $extraLoopCount minutes." >> ../log/watchdog/alive
            sleep 60
            let extraLoopCount=$(($extraLoopCount - 1))
        done
    else
        mv ../log/watchdog/alive ../log/watchdog/alive.old
        echo `date` > ../log/watchdog/alive
        echo "All is well." >> ../log/watchdog/alive

        sleep 60
    fi

    # check if we should try to clean up log files
    if [ $(($currentTime - $logfileCleanupTime)) -gt $cleanupTimeout ]
    then
        let logfileCleanupTime=$currentTime;

        ./removeOldLogFiles.sh
    fi
done
