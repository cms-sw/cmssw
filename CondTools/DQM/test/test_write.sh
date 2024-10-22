#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

if test -f "testXML.db"; then
    echo "==> removing pre-existing testXML.db"
    rm -f testXML.db
fi

echo "TESTING CondTools/DQM ..."
cmsRun ${SCRAM_TEST_PATH}/DQMUploadXMLFile.py || die "Failure running testCondToolsDQMUpload" $?
