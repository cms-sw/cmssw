#!/bin/tcsh

if ($#argv != 1) then
    echo "************** Argument Error: 1 arg. required **************"
    echo "   Usage:"
    echo "     ./dtNoiseProdChain.csh <run number>"
    echo "*************************************************************"
    exit 1
endif

set runn=$1
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

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh

cd $cmsswDir
eval `scramv1 runtime -csh`

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/noise/noise_${runn}.db ) ) then

    cd ${workDir}/Run${runn}/Noise/Production
    foreach lastCrab (crab_0_*)
	set crabDir=$lastCrab
    end

    if( ! ( -e ${crabDir}/res/noise_1.db ) ) then
	source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh
	crab -getoutput
    endif

    cd ${crabDir}/res
    if( ! -e ./noise_1.db ) then
	echo "WARNING: noise_1.db file not found! Exiting!"
	exit 1
    endif

    cp noise_1.db /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/noise/noise_${runn}.db

    # Move DTNoiseCalib_1.root to /afs/.../noise/ directory,
    # but do not exit if DTNoiseCalib_1.root was not produced
    if( -e ./DTNoiseCalib_1.root ) then
	cp DTNoiseCalib_1.root /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/noise/DTNoiseCalib_${runn}.root
    else
	echo "WARNING: DTNoiseCalib_1.root file not found!"
    endif

endif

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/noise/noise_${runn}.db ) ) then
    echo "File /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/noise/noise_${runn}.db not found!"
    echo "Exiting!"
    exit 1
endif

##
## DumpDBToFile.cfg
##

cd ${cmsswDir}/CalibMuon/DTCalibration/test
cat DumpDBToFile_noise_TEMPL_cfg.py | sed "s?RUNPERIODTEMPL?${runp}?g" | sed "s?RUNNUMBERTEMPLATE?${runn}?g"  >! DumpDBToFile_noise_${runn}_cfg.py


echo "Starting cmsRun DumpDBToFile_noise_${runn}_cfg.py"
cmsRun DumpDBToFile_noise_${runn}_cfg.py >&! tmpDumpDBToFile_noise_${runn}.log
echo "Finished cmsRun DumpDBToFile_noise_${runn}_cfg.py"

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/noise/noise_${runn}.txt ) ) then
    echo "DumpDBToFile_noise_`echo $runn`_cfg.py did not produce txt file!"
    echo "See tmpDumpDBToFile_noise_${runn}.log for details"
    echo "(in your CMSSW test directory)."
    exit 1
else
    rm DumpDBToFile_noise_${runn}_cfg.py tmpDumpDBToFile_noise_${runn}.log
endif


cd $workDir

echo "DT noise validation chain completed successfully!"
exit 0
