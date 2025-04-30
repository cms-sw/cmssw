#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/Zmumu single configuration with json..."
pushd test_yaml/Zmumu/single/testSingleZMM/unitTestZmumuMCreal/1/
./cmsRun validation_cfg.py config=validation.json || die "Failure running Zmumu first single configuration with json" $?
popd

echo "TESTING Alignment/Zmumu single configuration with json..."
pushd test_yaml/Zmumu/single/testSingleZMM/unitTestZmumuMCdesign/1/
./cmsRun validation_cfg.py config=validation.json || die "Failure running Zmumu second single configuration with json" $?

echo "TESTING Alignment/Zmumu single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running Zmumu single configuration standalone" $?
popd

echo "TESTING Zmumu merge step"
pushd test_yaml/Zmumu/merge/testSingleZMM/1/
./Zmumumerge --verbose validation.json || die "Failure running Zmumu merge step" $?
popd
-- dummy change --
