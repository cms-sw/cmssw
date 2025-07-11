#!/bin/bash

function die { echo $1: status $2 ; exit $2; }

runrange=376370-379254

thistest="Alignment/PixelBarycentre single yaml configuration unitTest (test 1/4)"
echo "TESTING $thistest..."
pushd test_yaml/PixBary/single/testSinglePixBary/unitTest/$runrange
./cmsRun validation_cfg.py config=validation.json || die "Failure running $thistest" $?
popd

thistest="Alignment/PixelBarycentre single yaml configuration mp3619 (test 2/4)"
echo "TESTING $thistest..."
pushd test_yaml/PixBary/single/testSinglePixBary/mp3619/$runrange
./cmsRun validation_cfg.py config=validation.json || die "Failure running $thistest" $?
popd

thistest="Alignment/PixelBarycentre extract yaml configuration unitTest (test 3/4)"
echo "TESTING $thistest..."
pushd test_yaml/PixBary/extract/testExtractPixBary/testSinglePixBary/unitTest/$runrange
./run.sh || die "Failure running $thistest" $?
popd

thistest="Alignment/PixelBarycentre extract yaml configuration mp3619 (test 4/4)"
echo "TESTING $thistest..."
pushd test_yaml/PixBary/extract/testExtractPixBary/testSinglePixBary/mp3619/$runrange
./run.sh || die "Failure running $thistest" $?
popd
