#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

#the last few lines of the output are the printout from the
# ConcurrentModuleTimer service detailing how much time was
# spent in 2,3 or 4 modules running simultaneously.
touch empty_file

(cmsRun ${LOCAL_TEST_DIR}/test_1_concurrent_lumi_cfg.py 2>&1) | tail -n 2 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file || die "Failure using test_1_concurrent_lumi_cfg.py" $?

(cmsRun ${LOCAL_TEST_DIR}/test_2_concurrent_lumis_cfg.py 2>&1) | tail -n 1 | grep -v ' 0 ' | grep -v 'e-' | diff - empty_file && die "Failure using test_2_concurrent_lumis_cfg.py" 1

exit 0
