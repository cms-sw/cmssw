#!/bin/sh
conddb --yes copy EcalTPGStripStatus_v3_hlt --destdb EcalTPGStripStatus_v3_hlt_O2OTEST.db --o2oTest
cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/copyBadStrip_cfg.py destinationDatabase=sqlite_file:EcalTPGStripStatus_v3_hlt_O2OTEST.db destinationTag=EcalTPGStripStatus_v3_hlt
ret=$?
conddb --db EcalTPGStripStatus_v3_hlt_O2OTEST.db list EcalTPGStripStatus_v3_hlt
echo "return code is $ret"
exit $ret
