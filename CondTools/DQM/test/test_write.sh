#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

if test -f "testXML.db"; then
    echo "==> removing pre-existing testXML.db"
    rm -f testXML.db
fi

echo "TESTING CondTools/DQM ..."
cmsRun ${LOCAL_TEST_DIR}/DQMUploadXMLFile.py || die "Failure running testCondToolsDQMUpload" $?
