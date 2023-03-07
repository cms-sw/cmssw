#!/bin/sh
conddb --yes copy EcalTPGLinearizationConst_v2_hlt --destdb EcalTPGLinearizationConst_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyLin_cfg.py destinationDatabase=sqlite_file:EcalTPGLinearizationConst_v2_hlt_O2OTEST.db destinationTag=EcalTPGLinearizationConst_v2_hlt
ret=$?
conddb --db EcalTPGLinearizationConst_v2_hlt_O2OTEST.db list EcalTPGLinearizationConst_v2_hlt
echo "return code is $ret"
exit $ret
