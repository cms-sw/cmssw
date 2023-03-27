#!/bin/sh
conddb --yes copy EcalTPGPedestals_v2_hlt --destdb EcalTPGPedestals_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyPed_cfg.py destinationDatabase=sqlite_file:EcalTPGPedestals_v2_hlt_O2OTEST.db destinationTag=EcalTPGPedestals_v2_hlt
ret=$?
conddb --db EcalTPGPedestals_v2_hlt_O2OTEST.db list EcalTPGPedestals_v2_hlt
echo "return code is $ret"
exit $ret
