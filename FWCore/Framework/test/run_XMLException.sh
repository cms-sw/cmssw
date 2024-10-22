#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR="${CMSSW_BASE}/src/FWCore/Framework/test"
cmsRun -j testXMLSafeException.xml ${LOCAL_TEST_DIR}/testXMLSafeException_cfg.py
xmllint testXMLSafeException.xml || die "cmsRun testXMLSafeException_cfg.py produced invalid XML job report" $?
