#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PrePoolInputTest_cfg.py || die 'Failure using PrePoolInputTest_cfg.py' $?

cp PoolInputTest.root PoolInputOther.root

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest_cfg.py || die 'Failure using PoolInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PrePool2FileInputTest_cfg.py || die 'Failure using PrePool2FileInputTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/Pool2FileInputTest_cfg.py || die 'Failure using Pool2FileInputTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PrePoolInputTest2_cfg.py || die 'Failure using PrePoolInputTest2_cfg.py' $?

cp PoolInputTest.root PoolInputOther.root

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest2_cfg.py || die 'Failure using PoolInputTest2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest3_cfg.py || die 'Failure using PoolInputTest3_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolEmptyTest_cfg.py || die 'Failure using PoolEmptyTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolEmptyTest2_cfg.py || die 'Failure using PoolEmptyTest2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/poolsource_multiprocess_gen_file_cfg.py || die 'Failure using poolsource_multiprocess_gen_file_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/poolsource_multiprocess_cfg.py || die 'Failure using poolsource_multiprocess_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/poolsource_multiprocess_gen_file_oneRun_cfg.py || die 'Failure using poolsource_multiprocess_gen_file_oneRun_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/poolsource_multiprocess_oneRun_cfg.py || die 'Failure using poolsource_multiprocess_oneRun_cfg.py' $?

#test reading of the old format files
for file in ${CMSSW_BASE}/src/IOPool/Input/testdata/old*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_old_formats_cfg.py "$file" || die 'Failed to read old file $file' $?
done

for file in ${CMSSW_BASE}/src/IOPool/Input/testdata/empty*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_empty_old_formats_cfg.py "$file" || die 'Failed to read old empty file $file' $?
done

# Note that the expected sequence of runs, lumis, and events will change slightly after 3_8_0 so
# a different test config will be required to run the following test. (The generation configs will
# still be OK and can be used as is)
cmsRun ${LOCAL_TEST_DIR}/test_complex_before_3_8_0_cfg.py ${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_2_2_13.root || die 'Failed to read old 2_2_13 file' $?
cmsRun ${LOCAL_TEST_DIR}/test_complex_before_3_8_0_cfg.py ${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_3_5_0.root || die 'Failed to read old 3_5_0 file' $?
cmsRun ${LOCAL_TEST_DIR}/test_complex_before_3_8_0_cfg.py ${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_3_7_0.root || die 'Failed to read old 3_7_0 file' $?

popd
