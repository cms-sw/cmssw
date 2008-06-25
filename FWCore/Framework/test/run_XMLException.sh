#!/bin/bash

#test=testRunMerge

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}
  echo ${LOCAL_TMP_DIR}
  cmsRun -j badXMLException.xml -p ${LOCAL_TEST_DIR}/testBadXMLJobReport.cfg
  xmllint badXMLException.xml || die "cmsRun testBadXMLJobReport.cfg produced invalid XML job report" $?

popd

exit 0
