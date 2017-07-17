#!/bin/sh


function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

rm -f goodDataFormatsFWLite.root good2DataFormatsFWLite.root emptyDataFormatsFWLite.root
rm -f other_onlyDataFormatsFWLite.root good_delta5DataFormatsFWLite.root
rm -f partialEventDataFormatsFWLite.root refTestCopyDropDataFormatsFWLite.root

echo RefTest_cfg.py
cmsRun ${LOCAL_TEST_DIR}/RefTest_cfg.py || die "cmsRun RefTest_cfg.py" $?
echo RefTest2_cfg.py
cmsRun ${LOCAL_TEST_DIR}/RefTest2_cfg.py || die "cmsRun RefTest2_cfg.py" $?
echo EmptyFile_cfg.py
cmsRun ${LOCAL_TEST_DIR}/EmptyFile_cfg.py || die "cmsRun EmptyFile_cfg.py" $?
echo PartialEventTest_cfg.py
cmsRun ${LOCAL_TEST_DIR}/PartialEventTest_cfg.py || die "cmsRun PartialEventTest_cfg.py" $?
echo RefTestCopyDrop_cfg.py
cmsRun ${LOCAL_TEST_DIR}/RefTestCopyDrop_cfg.py || die "cmsRun RefTestCopyDrop_cfg.py" $?

popd
exit 0
