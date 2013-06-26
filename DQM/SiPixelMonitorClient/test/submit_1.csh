#!/bin/csh
setenv startdir `pwd`
setenv mydir /afs/cern.ch/user/m/merkelp/scratch0/CMSSW_2_0_11/src/DQM/SiPixelCommon/test/55242_32
cd $mydir
cmsenv
setenv STAGE_SVCCLASS wan
cp Run_offline_DQM_33.cfg $startdir/.
cd $startdir
cmsRun Run_offline_DQM_33.cfg
cp *.root $mydir/.
cp *.log $mydir/.
