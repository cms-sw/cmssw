#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

export LOCAL_TEST_DIR=$CMSSW_BASE/src/DQMServices/Components/python/test/
echo ${LOCAL_TEST_DIR}

COLOR_WARN="\\033[0;31m"
COLOR_NORMAL="\\033[0;39m"
postfix=''

if [ $# -gt 0 ]; then
  postfix=$1;shift
fi

for bookIn in CTOR BJ BR
do
  if [[ "${bookIn}" != 'BR' && "${postfix}" == 'MultiThread' ]]; then
    continue
  fi
  echo -e "   ***   ${COLOR_WARN}RUNNING TEST 6 on ${bookIn} with OPTION $postfix${COLOR_NORMAL}   ***   "

  #OLD file1
  testConfig=create_file1_oldf_cfg.py
  rm -f dqm_file1_oldf.root
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} ${bookIn} ${postfix} &> /dev/null || die "cmsRun ${testConfig}" $?

  #OLD file3
  testConfig=create_file3_oldf_cfg.py
  rm -f dqm_file3_oldf.root
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} ${bookIn} ${postfix} &> /dev/null || die "cmsRun ${testConfig}" $?

  #check file1
  testConfig=check_file1_oldf_cfg.py 
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #check file3
  testConfig=check_file3_oldf_cfg.py
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #MERGE file1file3
  testConfig=merge_file1_file3_oldf_cfg.py
  rm -f dqm_merged_file1_file3_oldf.root
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #CHECK MERGED
  testConfig=check_merged_file1_file3_oldf_cfg.py
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #HARVEST MERGED FILE
  testConfig=harv_merged_file1_file3_oldf_cfg.py
  rm -f DQM_V0001_R00000000?__Test__Merged_File1_File3_oldf__DQM.root
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #HARVEST SINGLE FILES 
  testConfig=harv_file1_file3_oldf_cfg.py
  rm -f DQM_V0001_R00000000?__Test__File1_File3_oldf__DQM.root
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #CHECK HARVESTED FILE
  echo -e "COMPARING: ${COLOR_WARN}single vs merged${COLOR_NORMAL}"
  compare_using_files.py DQM_V0001_R000000001__Test__{,Merged_}File1_File3_oldf__DQM.root -C -s b2b -t 0.999999 2> /dev/null | grep 'Successes: 100.00%'
  compare_using_files.py DQM_V0001_R000000002__Test__{,Merged_}File1_File3_oldf__DQM.root -C -s b2b -t 0.999999 2> /dev/null | grep 'Successes: 100.00%'


done

exit 0
