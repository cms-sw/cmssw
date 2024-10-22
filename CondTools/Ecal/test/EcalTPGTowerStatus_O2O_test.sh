#!/bin/sh
conddb --yes copy EcalTPGTowerStatus_hlt --destdb EcalTPGTowerStatus_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyBadTT_cfg.py destinationDatabase=sqlite_file:EcalTPGTowerStatus_hlt_O2OTEST.db destinationTag=EcalTPGTowerStatus_hlt
ret=$?
conddb --db EcalTPGTowerStatus_hlt_O2OTEST.db list EcalTPGTowerStatus_hlt
echo "return code is $ret"
exit $ret
