#!/bin/bash
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }
(cmsRun ${SCRAM_TEST_PATH}/test_SiStripQualityStatistics_cfg.py) || die 'Failure running cmsRun test_SiStripQualityStatistics_cfg.py' $?
