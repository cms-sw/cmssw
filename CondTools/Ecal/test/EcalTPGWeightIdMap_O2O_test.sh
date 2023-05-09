#!/bin/sh
conddb --yes copy EcalTPGWeightIdMap_v2_hlt --destdb EcalTPGWeightIdMap_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyWIdMap_cfg.py destinationDatabase=sqlite_file:EcalTPGWeightIdMap_v2_hlt_O2OTEST.db destinationTag=EcalTPGWeightIdMap_v2_hlt
ret=$?
conddb --db EcalTPGWeightIdMap_v2_hlt_O2OTEST.db list EcalTPGWeightIdMap_v2_hlt
echo "return code is $ret"
exit $ret
