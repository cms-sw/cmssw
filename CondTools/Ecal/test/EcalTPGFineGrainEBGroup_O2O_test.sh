#!/bin/sh
conddb --yes copy EcalTPGFineGrainEBGroup_v2_hlt --destdb EcalTPGFineGrainEBGroup_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyFgrGroup_cfg.py destinationDatabase=sqlite_file:EcalTPGFineGrainEBGroup_v2_hlt_O2OTEST.db destinationTag=EcalTPGFineGrainEBGroup_v2_hlt
ret=$?
conddb --db EcalTPGFineGrainEBGroup_v2_hlt_O2OTEST.db list EcalTPGFineGrainEBGroup_v2_hlt
echo "return code is $ret"
exit $ret
