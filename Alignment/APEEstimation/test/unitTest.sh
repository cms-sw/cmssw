#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo " TESTING data set skimming"
# skim the predefined data set
python3 $CMSSW_BASE/src/Alignment/APEEstimation/test/SkimProducer/startSkim.py -s UnitTest || die "Failure skimming data set" $?

echo " TESTING auto submitter"
# start baseline measurement
python3 $CMSSW_BASE/src/Alignment/APEEstimation/test/autoSubmitter/autoSubmitter.py -c $CMSSW_BASE/src/Alignment/APEEstimation/test/autoSubmitter/unitTest.ini -u || die "Failure running autoSubmitter" $?

