#!/bin/bash
#set -x
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

function die { echo Failure $1: status $2 ; exit $2 ; }

logfile=testFrameworkExceptionHandling$1.log
cmsRun ${LOCAL_TEST_DIR}/testFrameworkExceptionHandling_cfg.py testNumber=$1 &> $logfile && die "cmsRun testFrameworkExceptionHandling_cfg.py testNumber=$1" 1

echo "There are two instances of ExceptionThrowingProducer with different module labels in this test."
echo "Usually one of them is configured to throw an intentional exception (an exception message should"
echo "always show up in the log file). The shell script and both modules run tests on how the Framework"
echo "handles the exception. The modules separately report in the log and here whether those tests PASSED"
echo "or FAILED."
echo ""

grep "ExceptionThrowingProducer FAILED" $logfile && die " - FAILED because found the following string in the log file: ExceptionThrowingProducer FAILED " 1
grep "ExceptionThrowingProducer PASSED" $logfile || die " - FAILED because cannot find the following string in the log file: ExceptionThrowingProducer PASSED " $?

# Look for the content that we expect to be in the exception messages.
# The first five should be the same in all log files.
# The other two or three are different for each case.

grep -q "Begin Fatal Exception" $logfile || die " - Cannot find the following string in the exception message: Begin Fatal Exception " $?

grep -q "An exception of category 'IntentionalTestException' occurred while" $logfile || die " - Cannot find the following string in the exception message: An exception of category 'IntentionalTestException' occurred while " $?

grep -q "Calling method for module ExceptionThrowingProducer/'throwException'" $logfile || die " - Cannot find the following string in the exception message: Calling method for module ExceptionThrowingProducer/'throwException' " $?

grep -q "Exception Message:" $logfile || die " - Cannot find the following string in the exception message: Exception Message: " $?

grep -q "End Fatal Exception" $logfile || die " - Cannot find the following string in the exception message: End Fatal Exception " $?

if [ $1 -eq 1 ]
then
    grep -q "Processing  Event run: 3 lumi: 1 event: 5" $logfile || die " - Cannot find the following string in the exception message: Processing  Event run: 3 lumi: 1 event: 5 " $?
    grep -q "Running path 'path1'" $logfile || die " - Cannot find the following string in the exception message: Running path 'path1' " $?
    grep -q "ExceptionThrowingProducer::produce, module configured to throw on: run: 3 lumi: 1 event: 5" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::produce, module configured to throw on: run: 3 lumi: 1 event: 5 " $?
fi

if [ $1 -eq 2 ]
then
    grep -q "Processing global begin Run run: 4" $logfile || die " - Cannot find the following string in the exception message: Processing global begin Run run: 4 " $?
    grep -q "ExceptionThrowingProducer::globalBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 3 ]
then
    grep -q "Processing global begin LuminosityBlock run: 4 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing global begin LuminosityBlock run: 4 luminosityBlock: 1 " $?
    grep -q "ExceptionThrowingProducer::globalBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0 " $?
fi

if [ $1 -eq 4 ]
then
    grep -q "Processing global end Run run: 3" $logfile || die " - Cannot find the following string in the exception message: Processing global end Run run: 3 " $?
    grep -q "ExceptionThrowingProducer::globalEndRun, module configured to throw on: run: 3 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalEndRun, module configured to throw on: run: 3 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 5 ]
then
    grep -q "Processing global end LuminosityBlock run: 3 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing global end LuminosityBlock run: 3 luminosityBlock: 1 " $?
    grep -q "ExceptionThrowingProducer::globalEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::globalEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0 " $?
fi

if [ $1 -eq 6 ]
then
    grep -q "Processing  stream begin Run run: 4" $logfile || die " - Cannot find the following string in the exception message: Processing  stream begin Run run: 4 " $?
    grep -q "ExceptionThrowingProducer::streamBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamBeginRun, module configured to throw on: run: 4 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 7 ]
then
    grep -q "Processing  stream begin LuminosityBlock run: 4 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing  stream begin LuminosityBlock run: 4 luminosityBlock: 1 " $?
    grep -q "ExceptionThrowingProducer::streamBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamBeginLuminosityBlock, module configured to throw on: run: 4 lumi: 1 event: 0 " $?
fi

if [ $1 -eq 8 ]
then
   grep -q "Processing  stream end Run run: 3" $logfile || die " - Cannot find the following string in the exception message: Processing  stream end Run run: 3 " $?
   grep -q "ExceptionThrowingProducer::streamEndRun, module configured to throw on: run: 3 lumi: 0 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamEndRun, module configured to throw on: run: 3 lumi: 0 event: 0 " $?
fi

if [ $1 -eq 9 ]
then
    grep -q "Processing  stream end LuminosityBlock run: 3 luminosityBlock: 1" $logfile || die " - Cannot find the following string in the exception message: Processing  stream end LuminosityBlock run: 3 luminosityBlock: 1 " $?
    grep -q "ExceptionThrowingProducer::streamEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::streamEndLuminosityBlock, module configured to throw on: run: 3 lumi: 1 event: 0 " $?
fi

if [ $1 -eq 10 ]
then
    grep -q "Processing begin Job" $logfile || die " - Cannot find the following string in the exception message: Processing begin Job " $?
    grep -q "ExceptionThrowingProducer::beginJob, module configured to throw during beginJob" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::beginJob, module configured to throw during beginJob " $?
fi

if [ $1 -eq 11 ]
then
    grep -q "Processing  begin Stream stream: 2" $logfile || die " - Cannot find the following string in the exception message: Processing  begin Stream stream: 2 " $?
    grep -q "ExceptionThrowingProducer::beginStream, module configured to throw during beginStream for stream: 2" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::beginStream, module configured to throw during beginStream for stream: 2 " $?
fi

if [ $1 -eq 12 ]
then
    grep -q "Processing begin ProcessBlock" $logfile || die " - Cannot find the following string in the exception message: Processing begin ProcessBlock " $?
    grep -q "ExceptionThrowingProducer::beginProcessBlock, module configured to throw during beginProcessBlock" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::beginProcessBlock, module configured to throw during beginProcessBlock " $?
fi

if [ $1 -eq 13 ]
then
    grep -q "Processing end ProcessBlock" $logfile || die " - Cannot find the following string in the exception message: Processing end ProcessBlock " $?
    grep -q "ExceptionThrowingProducer::endProcessBlock, module configured to throw during endProcessBlock" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::endProcessBlock, module configured to throw during endProcessBlock " $?
fi

if [ $1 -eq 14 ]
then
    grep -q "Processing  end Stream stream: 2" $logfile || die " - Cannot find the following string in the exception message: Processing  end Stream stream: 2 " $?
    grep -q "ExceptionThrowingProducer::endStream, module configured to throw during endStream for stream: 2" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::endStream, module configured to throw during endStream for stream: 2 " $?
fi

if [ $1 -eq 15 ]
then
    grep -q "Processing end Job" $logfile || die " - Cannot find the following string in the exception message: Processing end Job " $?
    grep -q "ExceptionThrowingProducer::endJob, module configured to throw during endJob" $logfile || die " - Cannot find the following string in the exception message: ExceptionThrowingProducer::endJob, module configured to throw during endJob " $?
fi

exit 0
