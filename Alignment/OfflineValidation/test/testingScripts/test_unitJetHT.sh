#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/JetHT single configuration with json..."
pushd test_yaml/JetHT/single/testJob/unitTestJetHT
./cmsRun validation_cfg.py config=validation.json || die "Failure running JetHT single configuration with json" $?

echo "TESTING Alignment/JetHT single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running JetHT single configuration standalone" $?
popd

echo "TESTING JetHT merge step"
pushd test_yaml/JetHT/merge/testJob/unitTestJetHT
./run.sh || die "Failure running JetHT merge step" $?
popd

echo "TESTING JetHT plotting"
pushd test_yaml/JetHT/plot/testJob/
./run.sh || die "Failure running JetHT plotting" $?
popd

echo "TESTING JetHT multi-IOV plotting"
jetHtPlotter $CMSSW_BASE/src/Alignment/OfflineValidation/test/examples/jetHt_multiYearTrendPlot.json || die "Failure running multi-IOV JetHT plotting" $?
