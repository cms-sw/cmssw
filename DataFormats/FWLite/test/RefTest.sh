#!/bin/sh


function die { echo Failure $1: status $2 ; exit $2 ; }


rm -f goodDataFormatsFWLite.root good2DataFormatsFWLite.root emptyDataFormatsFWLite.root
rm -f other_onlyDataFormatsFWLite.root good_delta5DataFormatsFWLite.root
rm -f partialEventDataFormatsFWLite.root refTestCopyDropDataFormatsFWLite.root

echo RefTest_cfg.py
cmsRun ${SCRAM_TEST_PATH}/RefTest_cfg.py || die "cmsRun RefTest_cfg.py" $?
echo RefTest2_cfg.py
cmsRun ${SCRAM_TEST_PATH}/RefTest2_cfg.py || die "cmsRun RefTest2_cfg.py" $?
echo EmptyFile_cfg.py
cmsRun ${SCRAM_TEST_PATH}/EmptyFile_cfg.py || die "cmsRun EmptyFile_cfg.py" $?
echo PartialEventTest_cfg.py
cmsRun ${SCRAM_TEST_PATH}/PartialEventTest_cfg.py || die "cmsRun PartialEventTest_cfg.py" $?
echo RefTestCopyDrop_cfg.py
cmsRun ${SCRAM_TEST_PATH}/RefTestCopyDrop_cfg.py || die "cmsRun RefTestCopyDrop_cfg.py" $?

exit 0
