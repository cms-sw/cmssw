#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo -e "TESTING Dropbox Metadata DB codes ...\n\n"
cd ${LOCAL_TEST_DIR}/
cmsRun ProduceDropBoxMetadata.py  || die "Failure running ProduceDropBoxMetadata.py" $? 

echo -e "TESTING Dropbox Metadata DB Reader code ...\n Reading local sqlite DropBoxMetadata.db \n\n "
cmsRun DropBoxMetadataReader.py || die "Failure running DropBoxMetadataReader.py" $?

