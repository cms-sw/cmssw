#!/usr/bin/env sh

./configureHosts.sh

source testSetup.sh

tests="\
 CommandQueue_t\
 ConcurrentQueue_t\
 ConsumerID_t\
 ConsumerRegistrationInfo_t\
 EnquingPolicyTag_t\
 MockNotifier_t\
 EventDistributor_t\
 EventQueueCollection_t\
 ExpirableQueue_t\
 FragmentStore_t\
 I2OChain_t\
 InitMsgCollection_t\
 MonitoredQuantity_t\
 QueueID_t\
 ResourceMonitorCollection_t\
 Sleep_t\
 StreamQueue_t\
 Time_t\
 TriggerSelector_t\
 state_machine_t\
 xhtmlmaker_t"

error=0

for test in $tests ; do
    startTime=`date "+%H:%M:%S"`
    printf "%-36s: $startTime " $test
    
    if $test > $test.log 2>&1
	then
	date "+%H:%M:%S Passed"
    else
	date "+%H:%M:%S FAILED"
	error=1
    fi
done

exit $error
