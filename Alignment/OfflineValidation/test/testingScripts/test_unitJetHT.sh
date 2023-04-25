#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/JetHT single configuration with json..."
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/JetHT/single/testJob/unitTestJetHT
./cmsRun validation_cfg.py config=validation.json || die "Failure running JetHT single configuration with json" $?

echo "TESTING Alignment/JetHT single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running JetHT single configuration standalone" $?

echo "TESTING JetHT merge step"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/JetHT/merge/testJob/unitTestJetHT
./run.sh || die "Failure running JetHT merge step" $?

echo "TESTING JetHT plotting"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/test_yaml/JetHT/plot/testJob/
./run.sh || die "Failure running JetHT plotting" $?

echo "TESTING JetHT multi-IOV plotting"
cd $CMSSW_BASE/src/Alignment/OfflineValidation/test/examples
jetHtPlotter jetHt_multiYearTrendPlot.json || die "Failure running multi-IOV JetHT plotting" $?
