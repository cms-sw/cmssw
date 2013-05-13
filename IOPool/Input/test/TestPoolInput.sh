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

cmsRun --parameter-set ${LOCAL_TEST_DIR}/poolsource_multiprocess_WithSkip_cfg.py || die 'Failure using poolsource_multiprocess_WithSkip_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/poolsource_multiprocess_selectevents_cfy.py || die 'Failure using poolsource_multiprocess_selectevents_cfy.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/poolsource_multiprocess_emptyrunslumis_cfy.py || die 'Failure using poolsource_multiprocess_emptyrunslumis_cfy.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep1_cfg.py || die 'Failure using PoolAliasTestStep1_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep2_cfg.py || die 'Failure using PoolAliasTestStep2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep2A_cfg.py || die 'Failure using PoolAliasTestStep2A_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasSubProcessTestStep1_cfg.py || die 'Failure using PoolAliasSubProcessTestStep1_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasSubProcessTestStep2_cfg.py || die 'Failure using PoolAliasSubProcessTestStep2_cfg.py' $?

#test merging of heterogeneous files with extra provenenace in subsequent files

cmsRun --parameter-set ${LOCAL_TEST_DIR}/preMerge_cfg.py || die 'Failure using preMerge_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/preMerge2_cfg.py || die 'Failure using preMerge2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/HeteroMerge_cfg.py || die 'Failure using HeteroMerge_cfg.py' $?

#test reading of the old format files

for file in ${CMSSW_BASE}/src/IOPool/Input/testdata/raw*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_old_raw_data_step1_cfg.py "$file" || die "Failed to read old raw data file $file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_old_raw_data_step2_cfg.py || die "Failed to read raw data file converted from $file" $?
  rm -fr converted.root
done

for file in ${CMSSW_BASE}/src/IOPool/Input/testdata/old*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_old_formats_cfg.py "$file" || die "Failed to read old file $file" $?
done

for file in ${CMSSW_BASE}/src/IOPool/Input/testdata/empty*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_empty_old_formats_cfg.py "$file" || die "Failed to read old empty file $file" $?
done

# Note that the expected sequence of runs, lumis, and events changed slightly at 3_8_0 so
# a different test config is required to run the following test for earlier releases. 
for file in ${CMSSW_BASE}/src/IOPool/Input/testdata/complex*.root
do
  case $file in
  "${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_2_2_13.root" | "${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_3_5_0.root" | "${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_3_7_0.root") 
  script=test_complex_before_3_8_0_cfg.py
  ;;
  *)
  script=test_complex_old_formats_cfg.py
  ;;
  esac
  cmsRun ${LOCAL_TEST_DIR}/$script "$file" || die "Failed to read old complex file $file" $?
done

cmsRun ${LOCAL_TEST_DIR}/test_merge_two_files.py ${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_4_2_7.root ${CMSSW_BASE}/src/IOPool/Input/testdata/complex_old_format_CMSSW_4_2_8.root || die 'Failure using test_merge_two_files.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_reduced_ProcessHistory_cfg.py merged_files.root || die 'Failure using test_reduced_ProcessHistory_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_reduced_ProcessHistory_dup_cfg.py merged_files.root || die 'Failure using test_reduced_ProcessHistory_dup_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_reduced_ProcessHistory_end_cfg.py merged_files.root || die 'Failure using test_reduced_ProcessHistory_end_cfg.py' $?

popd
