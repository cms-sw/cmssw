#!/bin/sh -ex
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

cmsRun -j PoolInputTest_jobreport.xml ${LOCAL_TEST_DIR}/PrePoolInputTest_cfg.py PoolInputTest.root 11 561 7 6 3 || die 'Failure using PrePoolInputTest_cfg.py' $?

cmsRun  -j PoolGuidTest_jobreport.xml ${LOCAL_TEST_DIR}/PoolGUIDTest_cfg.py file:PoolInputTest.root && die 'PoolGUIDTest_cfg.py PoolInputTest.root did not throw an exception' 1
GUID_EXIT_CODE=$(edmFjrDump --exitCode PoolGuidTest_jobreport.xml)
if [ "x${GUID_EXIT_CODE}" != "x8034" ]; then
    echo "Inconsistent GUID test reported exit code ${GUID_EXIT_CODE} which is different from the expected 8034"
    exit 1
fi
GUID_NAME=$(edmFjrDump --guid PoolInputTest_jobreport.xml).root
cp PoolInputTest.root ${GUID_NAME}
cmsRun ${LOCAL_TEST_DIR}/PoolGUIDTest_cfg.py file:${GUID_NAME} || die 'Failure using PoolGUIDTest_cfg.py ${GUID_NAME}' $?


cp PoolInputTest.root PoolInputOther.root

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest_cfg.py || die 'Failure using PoolInputTest_cfg.py' $?
cmsRun  ${LOCAL_TEST_DIR}/PoolInputTest_noDelay_cfg.py >& ${LOCAL_TMP_DIR}/PoolInputTest_noDelay_cfg.txt || die 'Failure using PoolInputTest_noDelay_cfg.py' $?
grep 'event delayed read from source' ${LOCAL_TMP_DIR}/PoolInputTest_noDelay_cfg.txt && die 'Failure in PoolInputTest_noDelay_cfg.py, found delay reads from source' 1
cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest_skip_with_failure_cfg.py || die 'Failure using PoolInputTest_skip_with_failure_cfg.py' $?
cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest_skipBadFiles_cfg.py  || die 'Failure using PoolInputTest_skipBadFiles_cfg.py ' $?

cmsRun ${LOCAL_TEST_DIR}/PrePool2FileInputTest_cfg.py || die 'Failure using PrePool2FileInputTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/Pool2FileInputTest_cfg.py || die 'Failure using Pool2FileInputTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PrePoolInputTest2_cfg.py || die 'Failure using PrePoolInputTest2_cfg.py' $?

cp PoolInputTest.root PoolInputOther.root

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest2_cfg.py || die 'Failure using PoolInputTest2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolInputTest3_cfg.py || die 'Failure using PoolInputTest3_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolEmptyTest_cfg.py || die 'Failure using PoolEmptyTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolEmptyTest2_cfg.py || die 'Failure using PoolEmptyTest2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep1_cfg.py || die 'Failure using PoolAliasTestStep1_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep2_cfg.py || die 'Failure using PoolAliasTestStep2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep1_DifferentOrder_cfg.py || die 'Failure using PoolAliasTestStep1_DifferentOrder_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep2_DifferentOrder_cfg.py || die 'Failure using PoolAliasTestStep2_DifferentOrder_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep2A_cfg.py || die 'Failure using PoolAliasTestStep2A_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep1C_cfg.py || die 'Failure using PoolAliasTestStep2A_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasTestStep2C_cfg.py || die 'Failure using PoolAliasTestStep2A_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasSubProcessTestStep1_cfg.py || die 'Failure using PoolAliasSubProcessTestStep1_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolAliasSubProcessTestStep2_cfg.py || die 'Failure using PoolAliasSubProcessTestStep2_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PrePoolInputTest_cfg.py RunPerLumiTest.root 50 1 25 1 5 || die 'Failure using PrePoolInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/RunPerLumiTest_cfg.py 25 >& ${LOCAL_TMP_DIR}/RunPerLumiTest.txt || die 'Failure using RunPerLumiTest_cfg.py' $?
grep 'record' ${LOCAL_TMP_DIR}/RunPerLumiTest.txt | cut -d ' ' -f 4-11 > ${LOCAL_TMP_DIR}/RunPerLumiTest.filtered.txt
diff ${LOCAL_TEST_DIR}/unit_test_outputs/RunPerLumiTest.filtered.txt ${LOCAL_TMP_DIR}/RunPerLumiTest.filtered.txt || die 'incorrect output using RunPerLumiTest_cfg.py' $? 

cmsRun ${LOCAL_TEST_DIR}/RunPerLumiTest_cfg.py 50 >& ${LOCAL_TMP_DIR}/tooManyLumis.txt && die 'RunPerLumiTest_cfg.py should have failed but did not' 1
grep "MismatchedInputFiles" ${LOCAL_TMP_DIR}/tooManyLumis.txt || die  'RunPerLumiTest_cfg.py should have failed but did not' $?

cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRunTest_cfg.py 'file:RunPerLumiTest.root' 25 1 25 1 5 || die 'Failure using firstLuminosityBlockForEachRunTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/PrePoolInputTest_cfg.py firstLumiTest1.root 25 1 100 1 5 || die 'Failure using PrePoolInputTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/PrePoolInputTest_cfg.py firstLumiTest2.root 25 1 100 6 5 || die 'Failure using PrePoolInputTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRunTest_cfg.py 'file:firstLumiTest1.root,file:firstLumiTest2.root' 50 1 25 1 5 || die 'Failure using firstLuminosityBlockForEachRunTest_cfg.py with 2 files' $?
cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRunTest_cfg.py 'file:firstLumiTest1.root,file:firstLumiTest2.root' 50 1 25 1 5 shareRun || die 'Failure using firstLuminosityBlockForEachRunTest_cfg.py with 2 files which share a run' $?
cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py 'file:firstLumiTest1.root' 2 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py with first lumi 2' $?
cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py 'file:firstLumiTest1.root' 4 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py with first lumi 4' $?
cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py 'file:firstLumiTest1.root' 3 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis_Test_cfg.py with first lumi 3' $?

cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py 'file:firstLumiTest1.root' 2 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py with first lumi 2' $?
cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py 'file:firstLumiTest1.root' 4 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py with first lumi 4' $?
cmsRun ${LOCAL_TEST_DIR}/firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py 'file:firstLumiTest1.root' 3 || die 'Failure using firstLuminosityBlockForEachRun_skipLumis2_Test_cfg.py with first lumi 3' $?


#test merging of heterogeneous files with extra provenenace in subsequent files

cmsRun --parameter-set ${LOCAL_TEST_DIR}/preMerge_cfg.py || die 'Failure using preMerge_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/preMerge2_cfg.py || die 'Failure using preMerge2_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/HeteroMerge_cfg.py || die 'Failure using HeteroMerge_cfg.py' $?

#test reading of the old format files
IOPoolInputData=$CMSSW_BASE/src
for dir in $(echo $CMSSW_SEARCH_PATH | tr : '\n') ;  do
  if [ -d ${dir}/IOPool/Input/data ] ; then
    IOPoolInputData=${dir}
    break
  fi
done

for file in ${IOPoolInputData}/IOPool/Input/data/raw*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_old_raw_data_step1_cfg.py "$file" || die "Failed to read old raw data file $file" $?
  cmsRun ${LOCAL_TEST_DIR}/test_old_raw_data_step2_cfg.py || die "Failed to read raw data file converted from $file" $?
  rm -fr converted.root
done

for file in ${IOPoolInputData}/IOPool/Input/data/old*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_old_formats_cfg.py "$file" || die "Failed to read old file $file" $?
done

for file in ${IOPoolInputData}/IOPool/Input/data/empty*.root
do
  cmsRun ${LOCAL_TEST_DIR}/test_empty_old_formats_cfg.py "$file" || die "Failed to read old empty file $file" $?
done

# Note that the expected sequence of runs, lumis, and events changed slightly at 3_8_0 so
# a different test config is required to run the following test for earlier releases. 
for file in ${IOPoolInputData}/IOPool/Input/data/complex*.root
do
  case $file in
  "${IOPoolInputData}/IOPool/Input/data/complex_old_format_CMSSW_2_2_13.root" | "${IOPoolInputData}/IOPool/Input/data/complex_old_format_CMSSW_3_5_0.root" | "${IOPoolInputData}/IOPool/Input/data/complex_old_format_CMSSW_3_7_0.root")
  script=test_complex_before_3_8_0_cfg.py
  ;;
  *)
  script=test_complex_old_formats_cfg.py
  ;;
  esac
  cmsRun ${LOCAL_TEST_DIR}/$script "$file" || die "Failed to read old complex file $file" $?
done

cmsRun ${LOCAL_TEST_DIR}/test_merge_two_files.py ${IOPoolInputData}/IOPool/Input/data/complex_old_format_CMSSW_4_2_7.root ${IOPoolInputData}/IOPool/Input/data/complex_old_format_CMSSW_4_2_8.root || die 'Failure using test_merge_two_files.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_reduced_ProcessHistory_cfg.py merged_files.root || die 'Failure using test_reduced_ProcessHistory_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_reduced_ProcessHistory_dup_cfg.py merged_files.root || die 'Failure using test_reduced_ProcessHistory_dup_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_reduced_ProcessHistory_end_cfg.py merged_files.root || die 'Failure using test_reduced_ProcessHistory_end_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_make_multi_lumi_cfg.py || die 'Failure using test_make_multi_lumi_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/test_read_multi_lumi_as_one_cfg.py || die 'Failure using test_read_multi_lumi_as_one_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/test_make_overlapping_lumis_cfg.py || die 'Failure using test_make_overlapping_lumis_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/test_read_overlapping_lumis_cfg.py || die 'Failure using test_read_overlapping_lumis_cfg.py' $?

popd
exit 0
