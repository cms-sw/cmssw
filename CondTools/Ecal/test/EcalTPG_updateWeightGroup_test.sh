#!/bin/sh

cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/updateTPGWeightGroup.py output=EcalTPGWeightGroup_TEST.db input=$CMSSW_BASE/src/CondTools/Ecal/data/EcalTPGWeightGroup_perstrip_test.txt filetype=txt outputtag=unittest
ret=$?
echo "return code is $ret"
exit $ret
