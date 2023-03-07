#!/bin/sh
conddb --yes copy EcalTPGWeightGroup_v2_hlt --destdb EcalTPGWeightGroup_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyWGroup_cfg.py destinationDatabase=sqlite_file:EcalTPGWeightGroup_v2_hlt_O2OTEST.db destinationTag=EcalTPGWeightGroup_v2_hlt
ret=$?
conddb --db EcalTPGWeightGroup_v2_hlt_O2OTEST.db list EcalTPGWeightGroup_v2_hlt
echo "return code is $ret"
exit $ret
