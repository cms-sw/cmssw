#!/bin/bash
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }
(cmsRun ${LOCAL_TEST_DIR}/testSSTGain_PCL_FromRECO_cfg.py 0) || die 'Failure running cmsRun testSSTGain_PCL_FromRECO_cfg.py 0' $?
(cmsRun ${LOCAL_TEST_DIR}/testSSTGain_HARVEST_FromRECO.py 0) || die 'Failure running cmsRun testSSTGain_HARVEST_FromRECO.py 0' $?