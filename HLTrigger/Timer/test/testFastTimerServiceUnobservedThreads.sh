#! /bin/bash
# The job emits signals from 3 threads that the FastTimerService has not seen (one per event): it must not crash, and
# the service must report them at the end of the job.

LOG=testFastTimerServiceUnobservedThreads.log
cmsRun ${SCRAM_TEST_PATH}/testFastTimerServiceUnobservedThreads_cfg.py > $LOG 2>&1
STATUS=$?
if [ $STATUS -ne 0 ]; then
  cat $LOG
  echo "cmsRun failed with status $STATUS"
  exit $STATUS
fi
if ! grep -q "3 threads were first seen" $LOG; then
  cat $LOG
  echo "the FastTimerService did not report the 3 unobserved threads"
  exit 1
fi
