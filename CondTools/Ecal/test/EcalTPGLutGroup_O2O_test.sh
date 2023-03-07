#!/bin/sh
conddb --yes copy EcalTPGLutGroup_v2_hlt --destdb EcalTPGLutGroup_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyLutGroup_cfg.py destinationDatabase=sqlite_file:EcalTPGLutGroup_v2_hlt_O2OTEST.db destinationTag=EcalTPGLutGroup_v2_hlt
ret=$?
conddb --db EcalTPGLutGroup_v2_hlt_O2OTEST.db list EcalTPGLutGroup_v2_hlt
echo "return code is $ret"
exit $ret
