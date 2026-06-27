#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/PV single configuration with json..."
pushd test_yaml/PV/single/TestMC/unitTest/1/
./cmsRun validation_cfg.py config=validation.json || die "Failure running PV single configuration with json" $?

echo "TESTING Alignment/PV single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running PV single configuration standalone" $?
popd

echo "TESTING PV merge step"
pushd test_yaml/PV/merge/TestMC/1/
./PVmerge validation.json --verbose || die "Failure running PV merge step" $?
popd

echo "TESTING PV trends"
pushd test_yaml/PV/trends/TestMC/
./PVtrends validation.json --verbose || die "Failure running PV trends" $?
popd
