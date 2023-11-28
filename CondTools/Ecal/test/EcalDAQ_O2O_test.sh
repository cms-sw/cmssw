#!/bin/sh
conddb --yes copy EcalDAQTowerStatus_online --destdb EcalDAQTowerStatus_online_O2OTEST.db --o2oTest
popconRun $CMSSW_BASE/src/CondTools/Ecal/python/EcalDAQ_popcon.py -d sqlite_file:EcalDAQTowerStatus_online_O2OTEST.db -t EcalDAQTowerStatus_online -c
ret=$?
conddb --db EcalDAQTowerStatus_online_O2OTEST.db list EcalDAQTowerStatus_online
echo "return code is $ret"
exit $ret
