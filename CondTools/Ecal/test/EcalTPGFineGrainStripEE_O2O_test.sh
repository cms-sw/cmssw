#!/bin/sh
conddb --yes copy EcalTPGFineGrainStripEE_v2_hlt --destdb EcalTPGFineGrainStripEE_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyLutGroup_cfg.py destinationDatabase=sqlite_file:EcalTPGFineGrainStripEE_v2_hlt_O2OTEST.db destinationTag=EcalTPGFineGrainStripEE_v2_hlt
ret=$?
conddb --db EcalTPGFineGrainStripEE_v2_hlt_O2OTEST.db list EcalTPGFineGrainStripEE_v2_hlt
echo "return code is $ret"
exit $ret
