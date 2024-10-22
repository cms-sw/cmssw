#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

echo cmsRun RefTest_cfg.py
cmsRun ${LOCAL_TEST_DIR}/RefTest_cfg.py || die 'Failed in RefTest_cfg.py' $?

echo  cmsRun AssociationMapTest_cfg.py
cmsRun ${LOCAL_TEST_DIR}/AssociationMapTest_cfg.py || die 'Failed in AssociationMapTest_cfg.py' $?

echo cmsRun AssociationMapReadTest_cfg.py
cmsRun ${LOCAL_TEST_DIR}/AssociationMapReadTest_cfg.py || die 'Failed in AssociationMapReadTest_cfg.py' $?
