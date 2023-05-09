#!/bin/sh
conddb --yes copy EcalLaserAPDPNRatios_prompt_v2 --destdb EcalLaserAPDPNRatios_prompt_v2_O2OTEST.db --o2oTest
popconRun $CMSSW_BASE/src/CondTools/Ecal/python/EcalLaser_prompt_popcon.py -d sqlite_file:EcalLaserAPDPNRatios_prompt_v2_O2OTEST.db -t EcalLaserAPDPNRatios_prompt_v2 -c
ret=$?
conddb --db EcalLaserAPDPNRatios_prompt_v2_O2OTEST.db list EcalLaserAPDPNRatios_prompt_v2
echo "return code is $ret"
exit $ret
