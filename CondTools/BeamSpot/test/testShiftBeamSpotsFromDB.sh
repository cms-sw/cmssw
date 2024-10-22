#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING BeamSpotOnline From DB shifting codes ..."
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineShifter_cfg.py inputTag=BeamSpotOnlineLegacy startRun=375491 startLumi=1 || die "Failure shifting payload from BeamSpotOnlineLegacy" $?
cmsRun ${SCRAM_TEST_PATH}/BeamSpotOnlineShifter_cfg.py inputTag=BeamSpotOnlineHLT startRun=375491 startLumi=1 inputRecord=BeamSpotOnlineHLTObjectsRcd  || die "Failure shifting payload from BeamSpotOnlineHLT" $?
