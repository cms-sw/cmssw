#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

mkdir -p SITECONF
mkdir -p SITECONF/local
mkdir -p SITECONF/local/JobConfig
mkdir -p SITECONF/DUMMY_CROSS_SITE

export SITECONFIG_PATH=${PWD}/SITECONF/local
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cp ${LOCAL_TEST_DIR}/sitelocalconfig/no_source/site-local-config.xml ${SITECONFIG_PATH}/JobConfig/
F1=${LOCAL_TEST_DIR}/test_sitelocalconfig_no_source_cfg.py
(cmsRun $F1 ) || die "Failure using $F1" $?

cp ${LOCAL_TEST_DIR}/sitelocalconfig/source/site-local-config.xml ${SITECONFIG_PATH}/JobConfig/
F2=${LOCAL_TEST_DIR}/test_sitelocalconfig_source_cfg.py
(cmsRun $F2 ) || die "Failure using $F2" $?

F3=${LOCAL_TEST_DIR}/test_sitelocalconfig_override_cfg.py
(cmsRun $F3 ) || die "Failure using $F3" $?

cp ${LOCAL_TEST_DIR}/sitelocalconfig/no_source/site-local-config.xml ${SITECONFIG_PATH}/JobConfig/
F3=${LOCAL_TEST_DIR}/test_sitelocalconfig_override_cfg.py
(cmsRun $F3 ) || die "Failure using $F3 with no-source site-local-config" $?

cp ${LOCAL_TEST_DIR}/sitelocalconfig/catalog/site-local-config.xml  ${SITECONFIG_PATH}/JobConfig/
cp ${LOCAL_TEST_DIR}/sitelocalconfig/catalog/local/storage.json ${SITECONFIG_PATH}/
cp ${LOCAL_TEST_DIR}/sitelocalconfig/catalog/dummycross/storage.json ${SITECONFIG_PATH}/../DUMMY_CROSS_SITE/
F4=${LOCAL_TEST_DIR}/test_sitelocalconfig_catalog_cfg.py
(cmsRun $F4 ) || die "Failure using $F4" $?

