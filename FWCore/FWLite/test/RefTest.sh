#!/bin/sh

function die { echo Failure $1: status $2 ; exit $2 ; }

rm -f ${SCRAM_TEST_PATH}/good.root
rm -f ${SCRAM_TEST_PATH}/good2.root
rm -f ${SCRAM_TEST_PATH}/good_delta5.root
rm -f good.root
rm -f good2.root
rm -f good_delta5.root

cmsRun ${SCRAM_TEST_PATH}/RefTest_cfg.py || die "cmsRun RefTest_cfg.py" $?

cmsRun ${SCRAM_TEST_PATH}/RefTest2_cfg.py || die "cmsRun RefTest2_cfg.py" $?
