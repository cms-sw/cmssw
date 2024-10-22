#!/bin/sh
conddb --yes copy EcalTPGSpike_v3_hlt --destdb EcalTPGSpike_v3_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copySpikeTh_cfg.py destinationDatabase=sqlite_file:EcalTPGSpike_v3_hlt_O2OTEST.db destinationTag=EcalTPGSpike_v3_hlt
ret=$?
conddb --db EcalTPGSpike_v3_hlt_O2OTEST.db list EcalTPGSpike_v3_hlt
echo "return code is $ret"
exit $ret
