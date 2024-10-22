#!/bin/sh
conddb --yes copy EcalDCSTowerStatus_online --destdb EcalDCSTowerStatus_online_O2OTEST.db --o2oTest
popconRun $CMSSW_BASE/src/CondTools/Ecal/python/EcalDCS_popcon.py -d sqlite_file:EcalDCSTowerStatus_online_O2OTEST.db -t EcalDCSTowerStatus_online -c
ret=$?
conddb --db EcalDCSTowerStatus_online_O2OTEST.db list EcalDCSTowerStatus_online
echo "return code is $ret"
exit $ret
