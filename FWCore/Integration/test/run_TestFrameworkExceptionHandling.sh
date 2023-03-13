#!/bin/bash
#set -x
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

function die { echo Failure $1: status $2 ; exit $2 ; }

logfile=testFrameworkExceptionHandling$1.log
echo $logfile
cmsRun ${LOCAL_TEST_DIR}/testFrameworkExceptionHandling_cfg.py testNumber=$1 &> $logfile && die "cmsRun testFrameworkExceptionHandling_cfg.py testNumber=$1" 1

# Look for the content that we expect to be in the exception messages.
# The first five should be the same in all log files.
# The other two or three are different for each case.

grep "Begin Fatal Exception" $logfile || die " - Cannot find the following string in the exception message: Begin Fatal Exception " $?

grep "An exception of category 'IntentionalTestException' occurred while" $logfile || die " - Cannot find the following string in the exception message: An exception of category 'IntentionalTestException' occurred while " $?

grep "Calling method for module ExceptionThrowingProducer/'throwException'" $logfile || die " - Cannot find the following string in the exception message: Calling method for module ExceptionThrowingProducer/'throwException' " $?

grep "Exception Message:" $logfile || die " - Cannot find the following string in the exception message: Exception Message: " $?

grep "End Fatal Exception" $logfile || die " - Cannot find the following string in the exception message: End Fatal Exception " $?

if [ $1 -eq 1 ]
then
    grep "Processing  Event run: 3 lumi: 1 event: 5" $logfile || die " - Cannot find the following string in the exception message: Processing  Event run: 3 lumi: 1 event: 5 " $?
    grep "Running path 'path1'" $logfile || die " - Cannot find the following string in the exception message: Running path 'path1' " $?
    grep "ExceptionThrowingProducer::produce, module configured to throw on: run: 3 lumi: 1 event: 5" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::produce, module configured to throw on: run: 3 lumi: 1 event: 5 " $?
fi

if [ $1 -eq 2 ]
then
    grep "Processing global begin Run run: 4" $logfile || die " - Cannot find the following string in the exception message: Processing global begin Run run: 4 " $?
    grep "ExceptionThrowingProducer::globalBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 3 ]
then
    grep "Processing global begin LuminosityBlock run: 4 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing global begin LuminosityBlock run: 4 luminosityBlock: 1 " $?
    grep "ExceptionThrowingProducer::globalBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0 " $?
fi

if [ $1 -eq 4 ]
then
    grep "Processing global end Run run: 3" $logfile || die " - Cannot find the following string in the exception message: Processing global end Run run: 3 " $?
    grep "ExceptionThrowingProducer::globalEndRun, module configured to throw on: run: 3 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalEndRun, module configured to throw on: run: 3 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 5 ]
then
    grep "Processing global end LuminosityBlock run: 3 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing global end LuminosityBlock run: 3 luminosityBlock: 1 " $?
    grep "ExceptionThrowingProducer::globalEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0 " $?
fi

if [ $1 -eq 6 ]
then
    grep "Processing  stream begin Run run: 4" $logfile || die " - Cannot find the following string in the exception message: Processing  stream begin Run run: 4 " $?
    grep "ExceptionThrowingProducer::streamBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 7 ]
then
    grep "Processing  stream begin LuminosityBlock run: 4 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing  stream begin LuminosityBlock run: 4 luminosityBlock: 1 " $?
    grep "ExceptionThrowingProducer::streamBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0 " $?
fi

if [ $1 -eq 8 ]
then
   grep "Processing  stream end Run run: 3" $logfile || die " - Cannot find the following string in the exception message: Processing  stream end Run run: 3 " $?
   grep "ExceptionThrowingProducer::streamEndRun, module configured to throw on: run: 3 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamEndRun, module configured to throw on: run: 3 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 9 ]
then
    grep "Processing  stream end LuminosityBlock run: 3 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing  stream end LuminosityBlock run: 3 luminosityBlock: 1 " $?
    grep "ExceptionThrowingProducer::streamEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0 " $?
fi

exit 0
