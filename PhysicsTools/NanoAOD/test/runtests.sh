#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

#to be enabled with the right files
#cmsDriver.py test80X -s NANO --mc --eventcontent NANOAODSIM --datatier NANO --filein /store/relval/CMSSW_8_0_0/RelValTTbar_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_v4-v1/10000/A65CD249-BFDA-E511-813A-0025905A6066.root    --conditions auto:run2_mc -n 100 --era Run2_2016,run2_miniAOD_80XLegacy || die 'Failure using cmsdriver 80X' $?
cmsDriver.py test92X -s NANO --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --filein /store/relval/CMSSW_9_2_12/RelValTTbar_13/MINIAODSIM/PU25ns_92X_upgrade2017_realistic_v11-v1/00000/080E2624-F59D-E711-ACEE-0CC47A7C35A4.root  --conditions auto:phase1_2017_realistic -n 100 --era Run2_2017,run2_nanoAOD_92X || die 'Failure using cmsdriver 92X' $?

cmsDriver.py test94X -s NANO --mc --eventcontent NANOAODSIM --datatier NANOAODSIM --filein /store/relval/CMSSW_9_4_0_pre1/RelValTTbar_13/MINIAODSIM/PU25ns_93X_mc2017_realistic_v3-v1/00000/92FD5642-509D-E711-ADAB-0025905B85C6.root    --conditions auto:phase1_2017_realistic -n 100 --era Run2_2017 || die 'Failure using cmsdriver 94X' $?


