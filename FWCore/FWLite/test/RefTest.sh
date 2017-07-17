#!/bin/sh

function die { echo Failure $1: status $2 ; exit $2 ; }

rm -f ${LOCAL_TEST_DIR}/good.root
rm -f ${LOCAL_TEST_DIR}/good2.root
rm -f ${LOCAL_TEST_DIR}/good_delta5.root
rm -f good.root
rm -f good2.root
rm -f good_delta5.root

cmsRun ${LOCAL_TEST_DIR}/RefTest_cfg.py || die "cmsRun RefTest_cfg.py" $?

cmsRun ${LOCAL_TEST_DIR}/RefTest2_cfg.py || die "cmsRun RefTest2_cfg.py" $?
