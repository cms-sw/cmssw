#!/bin/sh
conddb --yes copy EcalTPGCrystalStatus_v2_hlt --destdb EcalTPGCrystalStatus_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyBadXT_cfg.py destinationDatabase=sqlite_file:EcalTPGCrystalStatus_v2_hlt_O2OTEST.db destinationTag=EcalTPGCrystalStatus_v2_hlt
ret=$?
conddb --db EcalTPGCrystalStatus_v2_hlt_O2OTEST.db list EcalTPGCrystalStatus_v2_hlt
echo "return code is $ret"
exit $ret
