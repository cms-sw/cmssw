#!/bin/sh

cmsRun ./src/CondTools/Ecal/python/updateTPGOddWeightIdMap.py output=EcalTPGOddWeightIdMap_TEST.db input=./src/CondTools/Ecal/data/EcalTPGOddWeightIdMap_perstrip_test.txt filetype=txt outputtag=unittest
ret=$?
echo "return code is $ret"
exit $ret
