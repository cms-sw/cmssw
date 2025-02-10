#!/bin/sh -ex
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

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
