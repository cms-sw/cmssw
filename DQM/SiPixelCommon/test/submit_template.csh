#!/bin/csh
source /afs/cern.ch/cms/sw/cmsset_default.csh
setenv startdir `pwd`
setenv mydir /afs/cern.ch/user/k/krose/scratch0/CMSSW_2_1_8/src/DQM/SiPixelCommon/test/
cd $mydir
cmsenv
setenv STAGE_SVCCLASS wan
cp Run_offline_DQM_NUM_cfg.py $startdir
cd $startdir
cmsRun Run_offline_DQM_NUM_cfg.py
rfcp *.root ${startdir}/JOB_NUM/
cp *.log ${startdir}/JOB_NUM/
