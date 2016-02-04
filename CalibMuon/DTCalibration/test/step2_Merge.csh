#!/bin/tcsh

if ($#argv != 1) then
    echo "************** Argument Error: 1 arg. required **************"
    echo "   Usage:"
    echo "     ./dtTtrigProdChain.csh <run number>"
    echo "*************************************************************"
    exit 1
endif

set runn = $1

set runp = `tail -n +5 DBTags.dat | grep runperiod | awk '{print $2}'`
set cmsswarea = `tail -n +5 DBTags.dat | grep cmsswwa | awk '{print $2}'`
#set conddbversion = `tail +5 DBTags.dat | grep conddbvs | awk '{print $2}'`
set datasetpath = `tail -n +5 DBTags.dat | grep dataset | awk '{print $2}'`
set muondigi = `tail -n +5 DBTags.dat | grep dtDigi | awk '{print $2}'`
set email = `tail -n +5 DBTags.dat | grep email | awk '{print $2}'`
set globaltag=`tail -n +5 DBTags.dat | grep globaltag | awk '{print $2}'`

setenv workDir `pwd`
setenv cmsswDir "${HOME}/$cmsswarea"

unalias ls

cd $cmsswDir
eval `scramv1 runtime -csh`

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTTimeBoxes_${runn}.root ) ) then

    cd ${workDir}/Run${runn}/Ttrig/Production

    foreach lastCrab (crab_0_*)
	set crabDir=$lastCrab
    end

    if( ! ( -e ${crabDir}/res/DTTimeBoxes_${runn}.root ) ) then

	source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh
	crab -getoutput

	cd ${crabDir}/res

	if( ! -e ./DTTimeBoxes_1.root ) then
	    echo "WARNING: no DTTimeBoxes_xxx.root files found! Exiting!"
	    exit 1
	endif

	hadd DTTimeBoxes_${runn}.root DTTimeBoxes_*.root
	cp DTTimeBoxes_${runn}.root /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/
    endif

endif

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTTimeBoxes_${runn}.root ) ) then
    echo "File /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTTimeBoxes_${runn}.root not found!"
    echo "Exiting!"
    exit 1
endif

##
## DTTTrigWriter.cfg
##

cd ${cmsswDir}/CalibMuon/DTCalibration/test
cat DTTTrigWriter_TEMPL_cfg.py | sed "s?RUNPERIODTEMPL?${runp}?g"  | sed "s/GLOBALTAGTEMPLATE/${globaltag}/g" | sed "s?RUNNUMBERTEMPLATE?${runn}?g" >! DTTTrigWriter_${runn}_cfg.py

#cat DTTTrigWriter_TEMPL_cfg.py | sed "s?RUNPERIODTEMPL?${runp}?g"  | sed "s?CMSCONDVSTEMPLATE?${conddbversion}?g"| sed "s?TEMPLATE?${runn}?g" >! DTTTrigWriter_${runn}_cfg.py

echo "Starting cmsRun DTTTrigWriter_${runn}.cfg"
cmsRun DTTTrigWriter_${runn}_cfg.py >&! tmpDTTTrigWriter_${runn}.log
echo "Finished cmsRun DTTTrigWriter_${runn}.cfg"

if( ! ( -e /afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/ttrig_first_${runn}.db ) ) then
    echo "DTTTrigWriter_`echo $runn`.cfg did not produce txt file!"
    echo "See tmpDTTTrigWriter_`echo $runn`.log for details"
    echo "(in your CMSSW test directory)."
    exit 1
else
    rm tmpDTTTrigWriter_${runn}.log
    rm DTTTrigWriter_${runn}_cfg.py
endif

##
## DumpDBToFile_first.cfg
##

set dumpdb="first"
cat DumpDBToFile_ttrig_TEMPL_cfg.py | sed "s?DUMPDBTEMPL?${dumpdb}?g"  | sed "s?RUNPERIODTEMPL?${runp}?g"  | sed "s?RUNNUMBERTEMPLATE?${runn}?g" >! DumpDBToFile_${dumpdb}_${runn}_cfg.py

#cat DumpDBToFile_ttrig_TEMPL_cfg.py | sed "s?DUMPDBTEMPL?${dumpdb}?g"  | sed "s?RUNPERIODTEMPL?${runp}?g"  | sed "s?TEMPLATE?${runn}?g" >! DumpDBToFile_${dumpdb}_${runn}_cfg.py

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

cd ${workDir}/

set filename = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/${runp}/ttrig/DTTimeBoxes_${runn}.root"

echo ${filename} >! runname.txt

root -l -b -q mergeTimeBoxes.C

#192.135.19.251


exit 0
