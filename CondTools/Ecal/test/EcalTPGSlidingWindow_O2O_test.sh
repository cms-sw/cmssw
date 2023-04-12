#!/bin/sh
conddb --yes copy EcalTPGSlidingWindow_v2_hlt --destdb EcalTPGSlidingWindow_v2_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copySli_cfg.py destinationDatabase=sqlite_file:EcalTPGSlidingWindow_v2_hlt_O2OTEST.db destinationTag=EcalTPGSlidingWindow_v2_hlt
ret=$?
conddb --db EcalTPGSlidingWindow_v2_hlt_O2OTEST.db list EcalTPGSlidingWindow_v2_hlt
echo "return code is $ret"
exit $ret
