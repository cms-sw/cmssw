#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

echo LOCAL_TMP_DIR = ${LOCAL_TMP_DIR}

pushd ${LOCAL_TMP_DIR}
  testConfig=create_run_only_file_cfg.py
  rm -f dqm_run_only.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=read_run_only_file_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_run_only_file.py
  echo ${checkFile} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} || die "python3 ${checkFile}" $?

  testConfig=create_lumi_only_file_cfg.py
  rm -f dqm_lumi_only.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=read_lumi_only_file_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_lumi_only_file.py
  echo ${checkFile} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} || die "python3 ${checkFile}" $?

  testConfig=create_run_lumi_file_cfg.py
  rm -f dqm_run_lumi.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=read_run_lumi_file_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_run_lumi_file.py
  echo ${checkFile} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} dqm_run_lumi.root || die "python3 ${checkFile}" $?

  #read write
  testConfig=read_write_run_lumi_file_cfg.py
  rm -f dqm_run_lumi_copy.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_run_lumi_file.py
  echo ${checkFile} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} dqm_run_lumi_copy.root || die "python3 ${checkFile}" $?

  #more than one type
  testConfig=create_file_multi_types_cfg.py
  rm -f dqm_file_multi_types.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_multi_types.py
  fileToCheck=dqm_file_multi_types.root
  echo ${checkFile} ${fileToCheck} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} ${fileToCheck} || die "python3 ${checkFile} ${fileToCheck}" $?

  testConfig=copy_file_multi_types_cfg.py
  rm -f dqm_copy_multi_types.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_multi_types.py
  fileToCheck=dqm_copy_multi_types.root
  echo ${checkFile}  ${fileToCheck} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} ${fileToCheck} || die "python3 ${checkFile} ${fileToCheck}" $?

  #merging
  testConfig=create_file1_cfg.py
  rm -f dqm_file1.root
  rm -f dqm_file1_jobreport.xml
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} -j dqm_file1_jobreport.xml || die "cmsRun ${testConfig}" $?

  # test GUID here
  checkFile=check_guid_file1.py
  echo ${checkFile} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} || die "python3 ${checkFile}" $?

  testConfig=create_file2_cfg.py
  rm -f dqm_file2.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=read_file1_file2_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=create_file3_cfg.py
  rm -f dqm_file3.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=read_file1_file3_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=merge_file1_file2_cfg.py
  rm -f dqm_merged_file1_file2.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_merged_file1_file2.py
  echo ${checkFile} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} || die "python3 ${checkFile}" $?

  testConfig=read_merged_file1_file2_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=merge_file1_file3_file2_cfg.py
  rm -f dqm_merged_file1_file3_file2.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=read_merged_file1_file3_file2_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=create_one_run_one_lumi_run_only_file_cfg.py
  rm -f dqm_one_run_one_lumi_run_only.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?
  cp dqm_one_run_one_lumi_run_only.root dqm_one_run_one_lumi_run_only_2.root
 
  testConfig=merge_one_run_one_lumi_run_only_cfg.py
  rm -f dqm_merged_one_run_one_lumi_run_only.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=merge_file1_file3_file2_filterOnRun1_cfg.py
  rm -f dqm_merged_file1_file3_file2_filterOnRun1.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_merged_file1_file3_file2_filterOnRun1_cfg.py
  echo ${checkFile} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} || die "python3 ${checkFile}" $?

  testConfig=read_write_merged_file1_file3_file2_filterOnRun1_cfg.py
  rm -f dqm_merged_file1_file3_file2_filterOnRun1_copy.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  checkFile=check_merged_file1_file3_file2_filterOnRun1_copy_cfg.py
  fileToCheck=dqm_merged_file1_file3_file2_filterOnRun1_copy.root
  echo ${checkFile}  ${fileToCheck} ------------------------------------------------------------
  python3 ${LOCAL_TEST_DIR}/${checkFile} ${fileToCheck} || die "python3 ${checkFile} ${fileToCheck}" $?

  testConfig=create_file4_cfg.py
  rm -f dqm_file4.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=merge_file1_file3_file4_cfg.py
  rm -f dqm_merged_file1_file3_file4.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  testConfig=read_merged_file1_file3_file4_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

# empty
  testConfig=create_empty_file_cfg.py
  rm -f dqm_empty.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  stat dqm_empty.root && die "file created by cmsRun ${testConfig}" $?

  testConfig=read_missing_file_cfg.py
  echo ${testConfig} ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${testConfig} && die "cmsRun ${testConfig}" $?
  
popd

exit 0
