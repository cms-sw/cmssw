#!/bin/sh
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/updateADCToGeV_test.py
ret=$?
conddb --db EcalADCToGeV.db list EcalADCToGeV_test
echo "return code is $ret"
exit $ret
