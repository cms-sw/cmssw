#!/bin/tcsh

if ($#argv != 1) then
    echo "************** Argument Error: 1 arg. required **************"
    echo "   Usage:"
    echo "     ./submitNoiseProd.csh <run number>"
    echo "*************************************************************"
    exit 1
endif

set runsel=$1
set runn=`echo ${runsel}|cut -f1 -d','`


set runp=`tail +5 DBTags.dat | grep runperiod | awk '{print $2}'`
set cmsswarea=`tail +5 DBTags.dat | grep cmsswwa | awk '{print $2}'`
#set conddbversion=`tail +5 DBTags.dat | grep conddbvs | awk '{print $2}'`
set datasetpath=`tail +5 DBTags.dat | grep dataset | awk '{print $2}'`
#set t0db=`tail +5 DBTags.dat | grep t0 | awk '{print $2}'`
#set noisedb=`tail +5 DBTags.dat | grep noise | awk '{print $2}'`
#set ttrigdb=`tail +5 DBTags.dat | grep ttrig | awk '{print $2}'`
set globaltag=`tail +5 DBTags.dat | grep globaltag | awk '{print $2}'`
set muondigi=`tail +5 DBTags.dat | grep dtDigi | awk '{print $2}'`
set email=`tail +5 DBTags.dat | grep email | awk '{print $2}'`

setenv workDir `pwd`
setenv cmsswDir "${HOME}/$cmsswarea"

if( ! -d ./Run`echo $runn` ) then
    mkdir Run`echo $runn`
endif

if( ! -d ./Run`echo $runn`/Noise ) then
    mkdir Run`echo $runn`/Noise
endif

if( ! -d ./Run`echo $runn`/Noise/Production ) then
    mkdir Run`echo $runn`/Noise/Production
endif

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh

cd $cmsswDir
eval `scramv1 runtime -csh`

cd CalibMuon/DTCalibration/test
cat crab_noise_prod_TEMPL.cfg | sed "s?DATASETPATHTEMPLATE?${datasetpath}?g" | sed "s/RUNNUMBERTEMPLATE/${runn}/g" |sed "s/EMAILTEMPLATE/${email}/g" >! ${workDir}/Run${runn}/Noise/Production/crab.cfg
cat DTNoiseAnalyzer_TEMPL_cfg.py | sed "s/RUNNUMBERTEMPLATE/${runn}/g" | sed "s?RUNPERIODTEMPL?${runp}?g"| sed "s/DIGITEMPLATE/${muondigi}/g"| sed "s/GLOBALTAGTEMPLATE/${globaltag}/g" >! ${workDir}/Run${runn}/Noise/Production/DTNoiseAnalyzer_cfg.py


cd ${workDir}/Run${runn}/Noise/Production

source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh

crab -create -submit all
cd ${workDir}

exit 0
