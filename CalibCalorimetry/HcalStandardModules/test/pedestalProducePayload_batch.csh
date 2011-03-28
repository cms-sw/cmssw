#!/bin/csh

cmsrel CMSSW_3_11_0
cd CMSSW_3_11_0/src/
cmsenv
cvs co CalibCalorimetry/HcalStandardModules
scram b
cd CalibCalorimetry/HcalStandardModules/test

set localDir = ${1}
# copy the file from you local directory (need to be specified in ${1} - first parameter for this script)
cp ${localDir}/pedstxt.zip .

./pedestalProducePayload.csh > logfile_pedestalProducePayload.txt

cp test.db $localDir
cp logfile_pedestalProducePayload.txt $localDir
cp metadata.txt $localDir


