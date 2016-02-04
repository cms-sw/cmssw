#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}
  echo ${LOCAL_TMP_DIR}
  cmsRun -j testXMLSafeException.xml -p ${LOCAL_TEST_DIR}/testXMLSafeException_cfg.py
  xmllint testXMLSafeException.xml || die "cmsRun testXMLSafeException_cfg.py produced invalid XML job report" $?

popd

exit 0
