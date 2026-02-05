#!/bin/bash

set -exuo pipefail

function die { echo "Failure $1: status $2" ; it $2 ; }

makeJobReport > jobreport.xml 2>&1 || die "makeJobReport" $?

cat jobreport.xml | ${LOCALTOP}/src/FWCore/MessageService/test/validateXML.py || die "validateXML" $?

diff -u ${LOCALTOP}/src/FWCore/MessageService/test/unit_test_outputs/makeJobReport_expected.xml jobreport.xml || die "diff" $?

exit 0
