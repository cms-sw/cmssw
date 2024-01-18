#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

echo "write empty file"
cmsRun ${LOCAL_TEST_DIR}/makeEmptyRootFile.py || die "cmsRun makeEmptyRootFile.py" $?

echo "read empty file"
cmsRun ${LOCAL_TEST_DIR}/useEmptyRootFile.py || die "cmsRun useEmptyRootFile.py" $?

exit 0
