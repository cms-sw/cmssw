#!/bin/sh
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/updateIntercali_test.py
ret=$?
conddb --db EcalIntercalibConstants_test.db list EcalIntercalib_test
echo "return code is $ret"
exit $ret
