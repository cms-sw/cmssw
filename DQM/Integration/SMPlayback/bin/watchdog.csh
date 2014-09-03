#!/usr/bin/env csh

# initialize the base number of jobs
@ baseJobCount = 0

# initialize the forced restart time
@ forcedRestartTime = `date '+%s'`
@ restartTimeout = 604800  # one week (needs to be less than range in logfile cleanup script)

# initialize the logfile cleanup time
@ logfileCleanupTime = $forcedRestartTime
@ cleanupTimeout =  86400  # one day

# start an initial monitor consumer
(cd ../log/watchdog ; cmsRun ../../cfg/http_test.py) >& /dev/null &
sleep 10

while (1)
    # fetch the current time (seconds since 1970)
    @ currentTime = `date '+%s'`

    # touch the shared memory key file to keep it current
    touch $SMPB_SHM_KEYFILE

    # count the number of background jobs we have running
    rm -f jobList.tmp >& /dev/null
    jobs >& jobList.tmp
    @ jobCount = `cat jobList.tmp | wc -l`
    #echo "cnt = $jobCount"

    # check that the monitor consumer is still receiving events
    @ activeConsumerCount = `find ../log/watchdog -name '*.count' -mmin -3 -print | wc -l`
    #echo "active consumer count = $activeConsumerCount"

    # if the background job has died or if it is time to
    # force a restart, restart everything
    if (($jobCount - $baseJobCount) < 1 || \
        $activeConsumerCount < 1 || \
        ($currentTime - $forcedRestartTime) > $restartTimeout) then
        @ forcedRestartTime = $currentTime
        mv ../log/watchdog/alive ../log/watchdog/alive.old
        echo `date` > ../log/watchdog/alive
        echo "A restart is needed, working on that." >> ../log/watchdog/alive

        # restart the BU/FU/SM
        source ./restartEverything.csh

        # count the baseline number of background jobs
        rm -f jobList.tmp >& /dev/null
        jobs >& jobList.tmp
        @ baseJobCount = `cat jobList.tmp | wc -l`
        #echo "base = $baseJobCount"

        # start a fresh monitor consumer
        (cd ../log/watchdog ; cmsRun ../../cfg/http_test.py) >& /dev/null &

        # sleep for an extra time (to prevent rapid restarts in case of bugs)
        @ extraLoopCount = 10
        while ($extraLoopCount > 0)
            mv ../log/watchdog/alive ../log/watchdog/alive.old
            echo `date` > ../log/watchdog/alive
            echo "Restart attempted, will return to the normal checkup loop in $extraLoopCount minutes." >> ../log/watchdog/alive
            sleep 60
            @ extraLoopCount -= 1
        end
    else
        mv ../log/watchdog/alive ../log/watchdog/alive.old
        echo `date` > ../log/watchdog/alive
        echo "All is well." >> ../log/watchdog/alive

        sleep 60
    endif

    # check if we should try to clean up log files
    if (($currentTime - $logfileCleanupTime) > $cleanupTimeout) then
        @ logfileCleanupTime = $currentTime;

        ./removeOldLogFiles.sh
    endif
end


# for future reference...

#set TIMESTAMP = `date '+%Y%m%d%H%M%S'`
#((cmsRun ../cfg/http_test.py) > /dev/null) >& ../log/consumer/CONS${TIMESTAMP}.log &
#cd ../log/builderUnit
#xdaq.exe -p 50080 -c ../../cfg/sm_playback.xml >& BU${TIMESTAMP}.log &
#alias startConsumer "cd $rootDir/log/consumer; cmsRun ../../cfg/http_test.py"
