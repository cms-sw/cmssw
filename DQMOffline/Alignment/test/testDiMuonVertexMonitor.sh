#!/bin/bash

 function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

echo -e " Tesing on Z->mm \n\n"

cmsRun ${SCRAM_TEST_PATH}/DiMuonTkAlDQMValidator_cfg.py resonance=Z  || die "Failure using DiMuonTkAlDQMValidator_cfg.py resonance=Z" $?
cmsRun ${SCRAM_TEST_PATH}/DiMuonTkAlDQMHarvester_cfg.py resonance=Z || die "Failure using DiMuonTkAlDQMHarvester_cfg.py resonance=Z" $?

echo -e " Testing on J/psi -> mm \n\n"

cmsRun ${SCRAM_TEST_PATH}/DiMuonTkAlDQMValidator_cfg.py resonance=Jpsi || die "Failure using DiMuonTkAlDQMValidator_cfg.py resonance=Jpsi" $?
cmsRun ${SCRAM_TEST_PATH}/DiMuonTkAlDQMHarvester_cfg.py resonance=Jpsi || die "Failure using DiMuonTkAlDQMHarvester_cfg.py resonance=Jpsi" $?

echo -e " Testing on Upsilon -> mm \n\n"

cmsRun ${SCRAM_TEST_PATH}/DiMuonTkAlDQMValidator_cfg.py resonance=Upsilon || die "Failure using DiMuonTkAlDQMValidator_cfg.py resonance=Upsilon" $?
cmsRun ${SCRAM_TEST_PATH}/DiMuonTkAlDQMHarvester_cfg.py resonance=Upsilon || die "Failure using DiMuonTkAlDQMHarvester_cfg.py resonance=Upsilon" $?
