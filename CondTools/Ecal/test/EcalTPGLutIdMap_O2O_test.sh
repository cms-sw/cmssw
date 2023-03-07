#!/bin/sh
conddb --yes copy EcalTPGLutIdMap_v2_hlt --destdb EcalTPGLutIdMap_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyLutIdMap_cfg.py destinationDatabase=sqlite_file:EcalTPGLutIdMap_v2_hlt_O2OTEST.db destinationTag=EcalTPGLutIdMap_v2_hlt
ret=$?
conddb --db EcalTPGLutIdMap_v2_hlt_O2OTEST.db list EcalTPGLutIdMap_v2_hlt
echo "return code is $ret"
exit $ret
