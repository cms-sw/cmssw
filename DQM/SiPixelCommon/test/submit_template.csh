#!/bin/csh
source /afs/cern.ch/cms/sw/cmsset_default.csh
setenv startdir `pwd`
setenv mydir CFGDIR
cd $mydir
cmsenv
setenv STAGE_SVCCLASS wan
cp Run_offline_DQM_NUM_cfg.py $startdir
cd $startdir
cmsRun Run_offline_DQM_NUM_cfg.py
rfcp *.root ${mydir}/JOB_NUM/
cp *.log ${mydir}/JOB_NUM/
cp *.txt ${mydir}/JOB_NUM/
