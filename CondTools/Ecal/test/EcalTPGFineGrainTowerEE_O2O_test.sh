#!/bin/sh
conddb --yes copy EcalTPGFineGrainTowerEE_v2_hlt --destdb EcalTPGFineGrainTowerEE_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyFgrTowerEE_cfg.py destinationDatabase=sqlite_file:EcalTPGFineGrainTowerEE_v2_hlt_O2OTEST.db destinationTag=EcalTPGFineGrainTowerEE_v2_hlt
ret=$?
conddb --db EcalTPGFineGrainTowerEE_v2_hlt_O2OTEST.db list EcalTPGFineGrainTowerEE_v2_hlt
echo "return code is $ret"
exit $ret
