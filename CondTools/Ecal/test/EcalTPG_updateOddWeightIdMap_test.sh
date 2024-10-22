#!/bin/sh

cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/updateTPGOddWeightIdMap.py output=EcalTPGOddWeightIdMap_TEST.db input=$CMSSW_BASE/src/CondTools/Ecal/data/EcalTPGOddWeightIdMap_perstrip_test.txt filetype=txt outputtag=unittest
ret=$?
echo "return code is $ret"
exit $ret
