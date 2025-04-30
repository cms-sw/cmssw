#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/MTS single configuration with json..."
pushd test_yaml/MTS/single/testSingleMTS/PromptNewTemplate/1
./cmsRun validation_cfg.py config=validation.json || die "Failure running MTS single configuration with json" $?

echo "TESTING Alignment/MTS single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running MTS single configuration standalone" $?
popd

pushd test_yaml/MTS/single/testSingleMTS/mp3619/1
./cmsRun validation_cfg.py config=validation.json || die "Failure running MTS single configuration with json (part 2)" $?

echo "TESTING Alignment/MTS single configuration standalone..."
./cmsRun validation_cfg.py || die "Failure running MTS single configuration standalone (part 2)" $?
popd

echo "TESTING MTS merge step"
pushd test_yaml/MTS/merge/testSingleMTS/1
./MTSmerge validation.json --verbose || die "Failure running MTS merge step" $?
popd
-- dummy change --
