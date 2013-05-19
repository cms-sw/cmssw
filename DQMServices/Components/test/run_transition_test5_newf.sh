#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

export LOCAL_TEST_DIR=$CMSSW_BASE/src/DQMServices/Components/python/test/
echo ${LOCAL_TEST_DIR}

for bookIn in CTOR BJ BR
do
  #NEW file4
  testConfig=create_file4_cfg.py
  rm -f dqm_file4.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} $bookIn ForceReset || die "cmsRun ${testConfig}" $?

  #check file4
  testConfig=check_file4.py 
  echo ${testConfig} ------------------------------------------------------------
  python  ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  #HARVEST SINGLE FILES 
  testConfig=harv_file4_cfg.py
  rm -f DQM_V0001_R00000000?__Test__File4__DQM.root
  echo ${testConfig} ------------------------------------------------------------
  cmsRun ${LOCAL_TEST_DIR}/${testConfig} || die "cmsRun ${testConfig}" $?

  #CHECK HARVESTED FILE
  echo COMPARING with reference ------------------------------------------------------------
  compare_using_files.py -C -s b2b -t 0.999999 DQM_V0001_R000000001__Test__File4__DQM.root /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/DQM_V0001_R000000001__Test__File1_oldf_REFERENCE__DQM.root   
  compare_using_files.py -C -s b2b -t 0.999999 DQM_V0001_R000000002__Test__File4__DQM.root /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/DQM_V0001_R000000002__Test__File3_REFERENCE__DQM.root 
done

exit 0
