#!/usr/bin/env sh

./configureHosts.sh

source testSetup.sh

tests="\
 CommandQueue_t\
 ConcurrentQueue_t\
 DQMEventConsumerRegistrationInfo_t\
 EnquingPolicyTag_t\
 EventConsumerRegistrationInfo_t\
 EventDistributor_t\
 EventQueueCollection_t\
 EventStreamConfigurationInfo_t\
 ExpirableQueue_t\
 FragmentStore_t\
 I2OChain_t\
 MemoryChainDuplicate_t\
 MonitoredQuantity_t\
 QueueID_t\
 Sleep_t \
 state_machine_t\
 workloop_t\
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
