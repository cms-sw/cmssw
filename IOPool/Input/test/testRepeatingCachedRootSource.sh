#!/bin/sh -ex
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun -j PoolInputRepeatingSourceTest_jobreport.xml ${LOCALTOP}/src/IOPool/Input/test/PrePoolInputTest_cfg.py PoolInputRepeatingSource.root 11 561 7 6 3 || die 'Failure using PrePoolInputTest_cfg.py' $?

cmsRun ${LOCALTOP}/src/IOPool/Input/test/test_repeating_cfg.py || die 'Failed cmsRun test_repeating_cfg.py'
