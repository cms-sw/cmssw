#!/bin/tcsh

if ($#argv != 1) then
    echo "************** Argument Error: 1 arg. required **************"
    echo "   Usage:"
    echo "     ./submitTtrigResidCorr.csh <run number>"
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

unalias ls

cd $cmsswDir
eval `scramv1 runtime -csh`

echo "Start DT validation chain"

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTkFactValidation_${runn}.root ) ) then

    cd ${workDir}/Run${runn}/Ttrig/Validation

    foreach lastCrab (crab_0_*)
	set crabDir=$lastCrab
    end

    if( ! ( -e ${crabDir}/res/DTkFactValidation_${runn}.root ) ) then
	source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh
	crab -getoutput
    endif

    cd ${crabDir}/res
    if( ! -e ./residuals_1.root ) then
    	echo "WARNING: no residuals_xxx.root file found! Exiting!"
    	exit 1
    endif

    hadd DTkFactValidation_${runn}.root residuals_*.root
    if( ! ( -e DTkFactValidation_${runn}.root ) ) then
        echo "Could not produce DTkFactValidation_${runn}.root...exiting!"
        exit 1
    endif
    cp DTkFactValidation_${runn}.root /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/

endif

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTkFactValidation_${runn}.root ) ) then
    echo "File /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTkFactValidation_${runn}.root not found!"
    echo "Exiting!"
    exit 1
endif

##
##  Perform  the first Quality Checks
##  DTkFactValidation_2_${runn}_cfg.py
##

cd ${cmsswDir}/DQM/DTMonitorModule/test

cat DTkFactValidation_2_TEMPL_cfg.py | sed "s?RUNNUMBERTEMPLATE?${runn}?g"   | sed "s?RUNPERIODTEMPLATE?${runp}?g" >! DTkFactValidation_2_${runn}_cfg.py

echo "Starting cmsRun DTkFactValidation_2_${runn}.cfg"
cmsRun DTkFactValidation_2_${runn}_cfg.py >&! SummaryResiduals_${runn}.log
echo "Finished cmsRun DTkFactValidation_2_${runn}.cfg"

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/SummaryResiduals_${runn}.root ) ) then
    echo "DTkFactValidation_2_`echo $runn`.cfg did not produce root file!"
    echo "See SummaryResiduals_${runn}.log for details"
    echo "(in your CMSSW test directory)."
    exit 1
else
    rm DTkFactValidation_2_${runn}_cfg.py SummaryResiduals_${runn}.log
endif

cd $workDir

echo "DT validation chain completed successfully!"

echo "Start DT Residual Correction"

cd ${cmsswDir}/CalibMuon/DTCalibration/test

##                                                                                          
## DTTTrigResidualCorrection.cfg                                                           
##                                                                                          

cat DTTTrigResidualCorrection_TEMPL_cfg.py  | sed "s?GLOBALTAGTEMPLATE?${globaltag}?g" | sed "s?RUNNUMBERTEMPLATE?${runn}?g" | sed "s?RUNPERIODTEMPL?${runp}?g" >! DTTTrigResidualCorrection_${runn}_cfg.py

echo "Creating final db from empty31X.db"
cp empty31X.db /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/ttrig_ResidCorr_${runn}.db

echo "Starting cmsRun DTTTrigResidualCorrection_${runn}_cfg.py"               
cmsRun DTTTrigResidualCorrection_${runn}_cfg.py >&! /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTTTrigResidualCorrection_${runn}_log.txt   
echo "Finished cmsRun DTTTrigResidualCorrection_${runn}_cfg.py"                             
if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/ttrig_ResidCorr_${runn}.db ) ) then                                                                     
 echo "DTTTrigResidualCorrection_cfg.py did not produce ttrig corrected DataBase!"         
 echo "See DTTTrigResidualCorrection_${runn}_log.txt file for details"                    
 echo "(in /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/)."          
 echo "Exiting!"                                                                          
exit 1                                                                                   
else                                                                                        
rm DTTTrigResidualCorrection_${runn}_cfg.py                                              
endif

set dumpdb="ResidCorr"

########################################################

##
## DumpDBToFile_ResidCorr.cfg
##
if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/ttrig_ResidCorr_${runn}.txt ) ) then

cat DumpDBToFile_ttrig_TEMPL_cfg.py | sed "s?DUMPDBTEMPL?${dumpdb}?g"  | sed "s?RUNPERIODTEMPL?${runp}?g"  | sed "s?RUNNUMBERTEMPLATE?${runn}?g" >! DumpDBToFile_${dumpdb}_${runn}_cfg.py

echo "Starting cmsRun DumpDBToFile_${dumpdb}_${runn}.cfg"
cmsRun DumpDBToFile_${dumpdb}_${runn}_cfg.py >&! tmpDumpDBToFile_${dumpdb}_${runn}.log
echo "Finished cmsRun DumpDBToFile_${dumpdb}_${runn}.cfg"

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/ttrig_${dumpdb}_${runn}.txt ) ) then
    echo "DumpDBToFile_${dumpdb}_ttrig.cfg did not produce txt file!"
    echo "See tmpDumpDBToFile_${dumpdb}_`echo $runn`.log for details"
    echo "(in your CMSSW test directory)."
    exit 1
else 
    rm tmpDumpDBToFile_${dumpdb}_${runn}.log
    rm DumpDBToFile_${dumpdb}_${runn}_cfg.py
endif

else 
echo "Exiting! DumpDBToFile_ResidCorr.cfg"
endif

##
## Final Validation with Residuals; production of residuals
## DTkFactValidation_1_ResidCorr_TEMPL_cfg.py
##
cd $workDir

set dumpdb="ResidCorr"

cd $cmsswDir
eval `scramv1 runtime -csh`

echo "DT Residual Correction sarted!"

cd DQM/DTMonitorModule/test

cat crab_Valid_TEMPL.cfg | sed "s?DATASETPATHTEMPLATE?${datasetpath}?g" | sed "s/RUNNUMBERTEMPLATE/${runsel}/g" | sed "s/EMAILTEMPLATE/${email}/g" >! ${workDir}/Run${runn}/Ttrig/Validation/crab.cfg
cat DTkFactValidation_1_TEMPL_cfg.py | sed "s?DUMPDBTEMPL?${dumpdb}?g"| sed "s?GLOBALTAGTEMPLATE?${globaltag}?g" | sed "s/RUNNUMBERTEMPLATE/${runn}/g" | sed "s?RUNPERIODTEMPLATE?${runp}?g" >! ${workDir}/Run${runn}/Ttrig/Validation/DTkFactValidation_1_cfg.py

cd ${workDir}/Run${runn}/Ttrig/Validation

source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh

crab -create -submit all

cd $workDir

echo "DT Residual Correction completed successfully!"
exit 0
