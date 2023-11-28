#!/bin/sh
conddb --yes copy EcalTPGFineGrainEBIdMap_v2_hlt --destdb EcalTPGFineGrainEBIdMap_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyFgrIdMap_cfg.py destinationDatabase=sqlite_file:EcalTPGFineGrainEBIdMap_v2_hlt_O2OTEST.db destinationTag=EcalTPGFineGrainEBIdMap_v2_hlt
ret=$?
conddb --db EcalTPGFineGrainEBIdMap_v2_hlt_O2OTEST.db list EcalTPGFineGrainEBIdMap_v2_hlt
echo "return code is $ret"
exit $ret
