#!/bin/sh


function die { echo Failure $1: status $2 ; exit $2 ; }

rm -f ${LOCAL_TEST_DIR}/good.root ${LOCAL_TEST_DIR}/good2.root ${LOCAL_TEST_DIR}/empty.root ${LOCAL_TEST_DIR}/other_only.root ${LOCAL_TEST_DIR}/good_delta5.root
rm -f good.root good2.root empty.root other_only.root good_delta5.root

cmsRun ${LOCAL_TEST_DIR}/RefTest_cfg.py || die "cmsRun RefTest_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/RefTest2_cfg.py || die "cmsRun RefTest2_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/EmptyFile_cfg.py || die "cmsRun EmptyFile_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/PartialEventTest_cfg.py || die "cmsRun PartialEventTest_cfg.py" $?
cmsRun ${LOCAL_TEST_DIR}/RefTestCopyDrop_cfg.py || die "cmsRun RefTestCopyDrop_cfg.py" $?
