#!/bin/sh
conddb --yes copy EcalTPGPhysicsConst_v2_hlt --destdb EcalTPGPhysicsConst_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyPhysConst_cfg.py destinationDatabase=sqlite_file:EcalTPGPhysicsConst_v2_hlt_O2OTEST.db destinationTag=EcalTPGPhysicsConst_v2_hlt
ret=$?
conddb --db EcalTPGPhysicsConst_v2_hlt_O2OTEST.db list EcalTPGPhysicsConst_v2_hlt
echo "return code is $ret"
exit $ret
