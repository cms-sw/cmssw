#!/bin/bash
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }
(cmsRun ${SCRAM_TEST_PATH}/testSSTGain_PCL_FromRECO_cfg.py era=A) || die 'Failure running cmsRun testSSTGain_PCL_FromRECO_cfg.py era=A' $?
(cmsRun ${SCRAM_TEST_PATH}/testSSTGain_PCL_FromRECO_cfg.py era=B) || die 'Failure running cmsRun testSSTGain_PCL_FromRECO_cfg.py era=B' $?
(cmsRun ${SCRAM_TEST_PATH}/testSSTGain_HARVEST_FromRECO.py 0) || die 'Failure running cmsRun testSSTGain_HARVEST_FromRECO.py 0' $?
(cmsRun ${SCRAM_TEST_PATH}/testSSTGain_MultiRun_ALCAHARVEST.py globalTag=auto:run3_data_express) || die 'Failure running cmsRun testSSTGain_MultiRun_ALCAHARVEST.py 0' $?
