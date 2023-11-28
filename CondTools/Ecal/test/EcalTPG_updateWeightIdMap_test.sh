#!/bin/sh

cmsRun $CMSSW_BASE/src/CondTools/Ecal/python/updateTPGWeightIdMap.py output=EcalTPGWeightIdMap_TEST.db input=$CMSSW_BASE/src/CondTools/Ecal/data/EcalTPGWeightIdMap_perstrip_test.txt filetype=txt outputtag=unittest
ret=$?
echo "return code is $ret"
exit $ret
