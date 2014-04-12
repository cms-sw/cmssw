#!/bin/tcsh

echo  "Job started at "
\date;

if ($#argv < 1) then
    echo "************** Argument Error: at least 1 arg. required **************"
    echo "*   Usage:                                                           *"
    echo "*     ./PVValidationSubmitter.csh <TagFile.dat> <options>            *"
    echo "*                                                                    *"
    echo "*   Options:                                                         *"
    echo "*   --dryRun             do not submit the job to lxbatch            *"
    echo "**********************************************************************"
    exit 1
endif

set inputsource=$1
set options=$2

#setenv STAGE_SVCCLASS cmscaf 
source /afs/cern.ch/cms/caf/setup.csh

setenv CMSSW_VER 4_2_3
setenv CMSSW_DIR ${CMSSW_BASE}/src/Alignment/OfflineValidation/test
setenv LXBATCH_DIR `pwd`

cd $CMSSW_DIR
eval `scramv1 runtime -csh`
cd $LXBATCH_DIR

cp ${CMSSW_DIR}/${inputsource} .

set jobname=`more ${inputsource} | grep jobname | awk '{print $2}'`
set isda=`more ${inputsource} | grep isda | awk '{print $2}'`
set applybows=`more ${inputsource} | grep applybows | awk '{print $2}'`
set applycorrs=`more ${inputsource} | grep applycorrs | awk '{print $2}'`
set datasetpath=`more ${inputsource} | grep datasetpath | awk '{print $2}'`
set maxevents=`more ${inputsource} | grep maxevents | awk '{print $2}'`
set globaltag=`more ${inputsource} | grep globaltag | awk '{print $2}'`
set alignobj=`more ${inputsource} | grep alignobj | awk '{print $2}'`
set taggeom=`more ${inputsource} | grep taggeom | awk '{print $2}'`
set apeobj=`more ${inputsource} | grep apeobj | awk '{print $2}'`
set tagape=`more ${inputsource} | grep tagape | awk '{print $2}'`
set bowsobj=`more ${inputsource} | grep bowsobj | awk '{print $2}'`
set tagbows=`more ${inputsource} | grep tagbows | awk '{print $2}'`
set tracktype=`more ${inputsource} | grep tracktype | awk '{print $2}'`
set outfile=`more ${inputsource} | grep outfile | awk '{print $2}'`

echo "Starting to validate"
cp ${CMSSW_DIR}/PVValidation_TEMPL_cfg.py .
cat PVValidation_TEMPL_cfg.py | sed "s?ISDATEMPLATE?${isda}?g" | sed "s?APPLYBOWSTEMPLATE?${applybows}?g" | sed "s?EXTRACORRTEMPLATE?${applycorrs}?g" | sed "s?DATASETTEMPLATE?${datasetpath}?g" | sed "s?MAXEVENTSTEMPLATE?${maxevents}?g" | sed "s?GLOBALTAGTEMPLATE?${globaltag}?g" | sed "s?ALIGNOBJTEMPLATE?${alignobj}?g" | sed "s?GEOMTAGTEMPLATE?${taggeom}?g" | sed "s?APEOBJTEMPLATE?${apeobj}?g" | sed "s?ERRORTAGTEMPLATE?${tagape}?g" | sed "s?TRACKTYPETEMPLATE?${tracktype}?g" | sed "s?OUTFILETEMPLATE?${outfile}?g" >! ${jobname}_cfg
if(${applybows} == "True") then
    cat ${jobname}_cfg | sed "s?BOWSOBJECTTEMPLATE?${bowsobj}?g" | sed "s?BOWSTAGTEMPLATE?${tagbows}?g"  >! ${jobname}_cfg.py
    rm ${jobname}_cfg
else 
    mv ${jobname}_cfg ${jobname}_cfg.py
endif

cp ${CMSSW_DIR}/PVValidation_TEMPL.lsf .
cat PVValidation_TEMPL.lsf | sed  "s?JOBNAMETEMPLATE?${jobname}?g" | sed "s?OUTFILETEMPLATE?${outfile}?g" >! ${jobname}.lsf

if (! -d  ${CMSSW_DIR}/submittedCfg) then
    mkdir ${CMSSW_DIR}/submittedCfg
    cp ${jobname}_cfg.py  ${CMSSW_DIR}/submittedCfg
    cp ${jobname}.lsf     ${CMSSW_DIR}/submittedCfg
else
    echo "${CMSSW_DIR}/submittedCfg already exists"
    cp ${jobname}_cfg.py  ${CMSSW_DIR}/submittedCfg
    cp ${jobname}.lsf     ${CMSSW_DIR}/submittedCfg
endif

if(${options} != "--dryRun") then
 echo "cmsRun ${jobname}_cfg.py"
 cmsRun ${jobname}_cfg.py >& ${jobname}.out;

 echo "Content of working directory is: "
 \ls -lrt

 if (! -d  ${CMSSW_DIR}/test/PVValResults) then
     mkdir ${CMSSW_DIR}/PVValResults
     cp ${outfile}   ${CMSSW_DIR}/PVValResults
     cp ${jobname}.out ${CMSSW_DIR}/PVValResults 
 else
     cp ${outfile}   ${CMSSW_DIR}/PVValResults
     cp ${jobname}.out ${CMSSW_DIR}/PVValResults 
 endif
endif

echo  "Job ended at "
\date;

exit 0
