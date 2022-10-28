#!/bin/sh

cmsRun ./src/CondTools/Ecal/python/updateTPGWeightGroup.py output=EcalTPGWeightGroup_TEST.db input=./src/CondTools/Ecal/data/EcalTPGWeightGroup_perstrip_test.txt filetype=txt outputtag=unittest
ret=$?
echo "return code is $ret"
exit $ret
