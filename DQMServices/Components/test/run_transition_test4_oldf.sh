#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

export LOCAL_TEST_DIR=$CMSSW_BASE/src/DQMServices/Components/python/test/
echo ${LOCAL_TEST_DIR}

for bookIn in BR
do
  #OLD file4
  testConfig=create_file4_oldf_cfg.py
  rm -f dqm_file4_oldf.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} $bookIn  || die "cmsRun ${testConfig}" $?

  #check file4
  testConfig=check_file4_oldf_cfg.py 
  echo ${testConfig} ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  #HARVEST SINGLE FILES 
  testConfig=harv_file4_oldf_cfg.py
  rm -f DQM_V0001_R00000000?__Test__File4_oldf__DQM.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  #CHECK HARVESTED FILE
  echo COMPARING with reference ------------------------------------------------------------
  compare_using_files.py DQM_V0001_R000000001__Test__File4_oldf__DQM.root /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/DQM_V0001_R000000001__Test__File1_oldf_REFERENCE__DQM.root -C -s b2b -t 0.999999 
  compare_using_files.py DQM_V0001_R000000002__Test__File4_oldf__DQM.root /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/DQM_V0001_R000000002__Test__File3_REFERENCE__DQM.root  -C -s b2b -t 0.999999 

done

exit 0
