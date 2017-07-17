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

echo -e "   ***   ${COLOR_WARN}RUNNING TEST 5 on BeginRun with OPTION $postfix${COLOR_NORMAL}   ***   "
for bookIn in BR
do
  #OLD file4
  testConfig=create_file4_cfg.py
  rm -f dqm_file4.root
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} $bookIn ${postfix} ForceReset &> /dev/null || die "cmsRun ${testConfig}" $?

  #check file4
  testConfig=check_file4.py 
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  python ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #HARVEST SINGLE FILES 
  testConfig=harv_file4_cfg.py
  rm -f DQM_V0001_R00000000?__Test__File4__DQM.root
  echo -e "${COLOR_WARN}${testConfig}${COLOR_NORMAL}"
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} &> /dev/null || die "cmsRun ${testConfig}" $?

  #CHECK HARVESTED FILE
  echo -e "COMPARING with reference ${COLOR_WARN}DQM_V0001_R000000001__Test__File4__DQM.root${COLOR_NORMAL}"
  compare_using_files.py DQM_V0001_R000000001__Test__File4__DQM.root /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/DQM_V0001_R000000001__Test__File1_oldf_REFERENCE__DQM.root -C -s b2b -t 0.999999 2> /dev/null | grep 'Successes: 100.00%'
  echo -e "COMPARING with reference ${COLOR_WARN}DQM_V0001_R000000002__Test__File4__DQM.root${COLOR_NORMAL}"
  compare_using_files.py DQM_V0001_R000000002__Test__File4__DQM.root /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/DQM_V0001_R000000002__Test__File3_REFERENCE__DQM.root  -C -s b2b -t 0.999999 2> /dev/null | grep 'Successes: 100.00%'

done

exit 0
