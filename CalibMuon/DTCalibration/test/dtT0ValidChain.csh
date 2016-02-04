#!/bin/tcsh

if ($#argv != 1) then
    echo "************** Argument Error: 1 arg. required **************"
    echo "   Usage:"
    echo "     ./dtT0ValidChain.csh <run number>"
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
set reft0db=`tail +5 DBTags.dat | grep reft0 | awk '{print $2}'`

setenv workDir `pwd`
setenv cmsswDir "${HOME}/$cmsswarea"

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh

cd $cmsswDir
eval `scramv1 runtime -csh`

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/t0/t0_${runn}.db ) ) then

    cd ${workDir}/Run${runn}/T0/Production
    foreach lastCrab (crab_0_*)
	set crabDir=$lastCrab
    end

    if( ! ( -e ${crabDir}/res/t0_1.db ) ) then
	source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh
	crab -getoutput
    endif

    cd ${crabDir}/res
    if( ! -e ./t0_1.db ) then
	echo "WARNING: t0_1.db file not found! Exiting!"
	exit 1
    endif

    cp t0_1.db /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/t0/t0_${runn}.db

endif

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/t0/t0_${runn}.db ) ) then
    echo "File /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/t0/t0_${runn}.db not found!"
    echo "Exiting!"
    exit 1
endif

##
## DumpDBToFile.cfg
##

cd ${cmsswDir}/CalibMuon/DTCalibration/test
cat DumpDBToFile_t0_TEMPL_cfg.py  | sed "s?RUNPERIODTEMPL?${runp}?g" | sed "s?RUNNUMBERTEMPLATE?${runn}?g" >! DumpDBToFile_t0_${runn}_cfg.py

echo "Starting cmsRun DumpDBToFile_t0_${runn}.cfg"
cmsRun DumpDBToFile_t0_${runn}_cfg.py >&! tmpDumpDBToFile_t0_${runn}.log
echo "Finished cmsRun DumpDBToFile_t0_${runn}.cfg"

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/t0/t0_${runn}.txt ) ) then
    echo "DumpDBToFile_`echo $runn`.cfg did not produce txt file!"
    echo "See tmpDumpDBToFile_t0_${runn}.log for details"
    echo "(in your CMSSW test directory)."
    exit 1
else
    rm DumpDBToFile_t0_${runn}_cfg.py tmpDumpDBToFile_t0_${runn}.log
endif


##  
## DTt0DBValidation_ORCONsqlite.cfg
##
echo "DT DQMOffline validation started"
cd ${cmsswDir}/DQMOffline/CalibMuon/test
cat DTt0DBValidation_TEMPL_cfg.py | sed "s?REFT0TEMPLATE?${reft0db}?g" | sed "s?RUNNUMBERTEMPLATE?${runn}?g" | sed "s?RUNPERIODTEMPLATE?${runp}?g" >!  DTt0DBValidation_${runn}_cfg.py
cmsRun DTt0DBValidation_${runn}_cfg.py >&! DTtTrigDBValidation_${runn}.log

mv t0DBMonitoring_${runn}.root /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/

echo "DT DQMOffline validation completed successfully!"


# cat DTt0DBValidation_ORCONsqlite_TEMPL.cfg | sed s/TZEROTEMPLATE/${t0db}/g | sed s/RUNNTEMPLATE/${runn}/g >! DTt0DBValidation_ORCONsqlite_${runn}.cfg

# echo "Starting cmsRun DTt0DBValidation_ORCONsqlite_${runn}.cfg"
# cmsRun DTt0DBValidation_ORCONsqlite_${runn}.cfg >&! /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/CRUZET/t0/DTt0DBValidation_${runn}_log.txt
# echo "Finished cmsRun DTt0Validation_ORCONsqlite_${runn}.cfg"

# if( ! -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/CRUZET/t0/DTt0DBValidation_${runn}.root ) then
#    echo "DTt0DBValidation_ORCONsqlite_${runn}.cfg did not produce root file!"
#    echo "See DTt0DBValidation_${runn}_log.txt for details"
#    echo "(in /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/CRUZET/t0/)."
#    exit 1
#else
#    rm DTt0DBValidation_ORCONsqlite_${runn}.cfg
# endif

cd $workDir

echo "DT t0 validation chain completed successfully!"
exit 0
