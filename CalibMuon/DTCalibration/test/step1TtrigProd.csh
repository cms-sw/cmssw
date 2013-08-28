#!/bin/tcsh

if ($#argv != 1) then
    echo "************** Argument Error: 1 arg. required **************"
    echo "   Usage:"
    echo "     ./submitTtrigProd.csh <run number>"
    echo "*************************************************************"
    exit 1
endif

set runsel=$1 
set runn=`echo ${runsel}|cut -f1 -d','`

set runp=`tail -n +5 DBTags.dat | grep runperiod | awk '{print $2}'`
set cmsswarea=`tail -n +5 DBTags.dat | grep cmsswwa | awk '{print $2}'`
set datasetpath=`tail -n +5 DBTags.dat | grep dataset | awk '{print $2}'`
set globaltag=`tail -n +5 DBTags.dat | grep globaltag | awk '{print $2}'`
set muondigi=`tail -n +5 DBTags.dat | grep dtDigi | awk '{print $2}'`
set email=`tail -n +5 DBTags.dat | grep email | awk '{print $2}'`

setenv workDir `pwd`
setenv cmsswDir "${HOME}/$cmsswarea"

if( ! -d ./Run`echo $runn` ) then
    mkdir Run`echo $runn`
endif

if( ! -d ./Run`echo $runn`/Ttrig ) then
    mkdir Run`echo $runn`/Ttrig
endif

if( ! -d ./Run`echo $runn`/Ttrig/Production ) then
    mkdir Run`echo $runn`/Ttrig/Production
endif

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh

cd $cmsswDir
eval `scramv1 runtime -csh`

cd CalibMuon/DTCalibration/test
cat crab_ttrig_prod_TEMPL.cfg | sed "s?DATASETPATHTEMPLATE?${datasetpath}?g" | sed "s/RUNNUMBERTEMPLATE/${runsel}/g" |sed "s/EMAILTEMPLATE/${email}/g" >! ${workDir}/Run${runn}/Ttrig/Production/crab.cfg
cat DTTTrigCalibration_TEMPL_cfg.py | sed "s/DIGITEMPLATE/${muondigi}/g" | sed "s/GLOBALTAGTEMPLATE/${globaltag}/g" >! ${workDir}/Run${runn}/Ttrig/Production/DTTTrigCalibration_cfg.py

cd ${workDir}/Run${runn}/Ttrig/Production

source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh

crab -create -submit all
cd ${workDir}

exit 0
