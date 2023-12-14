#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/DiMuonV single configuration with json..."
pushd test_yaml/DiMuonV/single/testUnits/unitTestDiMuonVMC/1/
./cmsRun validation_cfg.py config=validation.json || die "Failure running DiMuonV single configuration with json" $?

echo "TESTING Alignment/DiMuonV single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running DiMuonV single configuration standalone" $?
popd

echo "TESTING PV merge step"
pushd test_yaml/DiMuonV/merge/testUnits/1/
./DiMuonVmerge validation.json --verbose || die "Failure running DiMuonV merge step" $?
popd
