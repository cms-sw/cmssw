#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

echo testOptions1_cfg.py
cmsRun -p ${LOCAL_TEST_DIR}/testOptions1_cfg.py >& testOptions1.log || die "cmsRun testOptions1_cfg.py" $?
grep "Number of Streams = 1" testOptions1.log || die "Failed number of streams test" $?
grep "Number of Concurrent Lumis = 1" testOptions1.log || die "Failed number of concurrent lumis test" $?
grep "Number of Concurrent IOVs = 1" testOptions1.log || die "Failed number of concurrent IOVs test" $?

echo testOptions2_cfg.py
cmsRun -p ${LOCAL_TEST_DIR}/testOptions2_cfg.py >& testOptions2.log || die "cmsRun testOptions2_cfg.py" $?
grep "Number of Streams = 5" testOptions2.log || die "Failed number of streams test" $?
grep "Number of Concurrent Lumis = 4" testOptions2.log || die "Failed number of concurrent lumis test" $?
grep "Number of Concurrent IOVs = 3" testOptions2.log || die "Failed number of concurrent IOVs test" $?

echo testOptions3_cfg.py
cmsRun -p ${LOCAL_TEST_DIR}/testOptions3_cfg.py >& testOptions3.log || die "cmsRun testOptions3_cfg.py" $?
grep "Number of Streams = 6" testOptions3.log || die "Failed number of streams test" $?
grep "Number of Concurrent Lumis = 2" testOptions3.log || die "Failed number of concurrent lumis test" $?
grep "Number of Concurrent IOVs = 2" testOptions3.log || die "Failed number of concurrent IOVs test" $?

echo testOptions4_cfg.py
cmsRun -p ${LOCAL_TEST_DIR}/testOptions4_cfg.py >& testOptions4.log || die "cmsRun testOptions4_cfg.py" $?
grep "Number of Streams = 6" testOptions4.log || die "Failed number of streams test" $?
grep "Number of Concurrent Lumis = 6" testOptions4.log || die "Failed number of concurrent lumis test" $?
grep "Number of Concurrent IOVs = 6" testOptions4.log || die "Failed number of concurrent IOVs test" $?

echo testOptions5_cfg.py
cmsRun -p ${LOCAL_TEST_DIR}/testOptions5_cfg.py >& testOptions5.log || die "cmsRun testOptions5_cfg.py" $?
grep "Number of Streams = 1" testOptions5.log || die "Failed number of streams test" $?
grep "Number of Concurrent Lumis = 1" testOptions5.log || die "Failed number of concurrent lumis test" $?
grep "Number of Concurrent IOVs = 1" testOptions5.log || die "Failed number of concurrent IOVs test" $?

echo testOptions6_cfg.py
cmsRun -p ${LOCAL_TEST_DIR}/testOptions6_cfg.py >& testOptions6.log || die "cmsRun testOptions6_cfg.py" $?
# Cannot run the grep tests because by default the options are not dumped.
# You can however run this manually with a debugger and check (which was done)
# And also just run it and see it doesn't crash...

rm testOptions1.log
rm testOptions2.log
rm testOptions3.log
rm testOptions4.log
rm testOptions5.log
rm testOptions6.log

popd

exit 0
