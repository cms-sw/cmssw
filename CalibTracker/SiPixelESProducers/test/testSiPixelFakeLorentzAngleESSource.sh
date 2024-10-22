#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Fake Pixel Conditions Sources ..."

printf "TESTING SiPixelFakeLorentzAngleESSource (BPix) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testSiPixelFakeLorentzAngleESSource_cfg.py isBPix=True  || die "Failure testing SiPixelFakeLorentzAngleESSource (BPix)" $?

printf "TESTING SiPixelFakeLorentzAngleESSource (FPix) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testSiPixelFakeLorentzAngleESSource_cfg.py isFPix=True  || die "Failure testing SiPixelFakeLorentzAngleESSource (FPix)" $?

printf "TESTING SiPixelFakeLorentzAngleESSource (By Module) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testSiPixelFakeLorentzAngleESSource_cfg.py isByModule=True  || die "Failure testing SiPixelFakeLorentzAngleESSource (By Module)" $?

printf "TESTING SiPixelFakeLorentzAngleESSource (Phase-2) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testSiPixelFakeLorentzAngleESSource_cfg.py isPhase2=True  || die "Failure testing SiPixelFakeLorentzAngleESSource (Phase-2) " $?
