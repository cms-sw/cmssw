#!/bin/tcsh

echo  "Job started at "
\date;

if ($#argv < 2) then
    echo "************** Argument Error: at least 2 arg. required **************"
    echo "*   Usage:                                                           *"
    echo "*     ./PVValidationSubmitter.csh <TagFile.dat> <taskname> <options> *"
    echo "*                                                                    *"
    echo "*   Options:                                                         *"
    echo "*   --dryRun             do not submit the job to lxbatch            *"
    echo "**********************************************************************"
    exit 1
endif

set inputsource=$1
set taskname=$2
set options=$3

echo "Submitting validation for file $inputsource with $options in task $taskname"

source /afs/cern.ch/cms/caf/setup.csh

setenv CMSSW_DIR ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/
setenv LXBATCH_DIR `pwd`

cd $CMSSW_DIR
eval `scramv1 runtime -csh`
cd $LXBATCH_DIR

cp ${inputsource} .

set jobname=`more ${inputsource} | grep jobname | awk '{print $2}'`
set isda=`more ${inputsource} | grep isda | awk '{print $2}'`
set ismc=`more ${inputsource} | grep ismc | awk '{print $2}'`
set runcontrol=`more ${inputsource} | grep runcontrol | awk '{print $2}'`
set runnumber=`more ${inputsource} | grep runnumber | awk '{print $2}'` 
set applybows=`more ${inputsource} | grep applybows | awk '{print $2}'`
set applycorrs=`more ${inputsource} | grep applycorrs | awk '{print $2}'`
set filesource=`more ${inputsource} | grep filesource | awk '{print $2}'`
set datasetpath=`more ${inputsource} | grep datasetpath | awk '{print $2}'`
set maxevents=`more ${inputsource} | grep maxevents | awk '{print $2}'`
set globaltag=`more ${inputsource} | grep globaltag | awk '{print $2}'`
set allFromGT=`more ${inputsource} | grep allFromGT | awk '{print $2}'`
set alignobj=`more ${inputsource} | grep alignobj | awk '{print $2}'`
set taggeom=`more ${inputsource} | grep taggeom | awk '{print $2}'`
set apeobj=`more ${inputsource} | grep apeobj | awk '{print $2}'`
set tagape=`more ${inputsource} | grep tagape | awk '{print $2}'`
set bowsobj=`more ${inputsource} | grep bowsobj | awk '{print $2}'`
set tagbows=`more ${inputsource} | grep tagbows | awk '{print $2}'`
set tracktype=`more ${inputsource} | grep tracktype | awk '{print $2}'`
set vertextype=`more ${inputsource} | grep vertextype | awk '{print $2}'`
set lumilist=`more ${inputsource} | grep lumilist | awk '{print $2}'`
set ptcut=`more ${inputsource} | grep ptcut | awk '{print $2}'`
set outfile=`more ${inputsource} | grep outfile | awk '{print $2}'`

if(${ismc} == "") then 
    set ${ismc} = "False"
endif

cp ${CMSSW_DIR}/PVValidation_TEMPL_cfg.py .
cat PVValidation_TEMPL_cfg.py | sed "s?ISDATEMPLATE?${isda}?g" | sed "s?ISMCTEMPLATE?${ismc}?g" | sed "s?RUNCONTROLTEMPLATE?${runcontrol}?g"  | sed "s?RUNBOUNDARYTEMPLATE?${runnumber}?g" | sed "s?APPLYBOWSTEMPLATE?${applybows}?g" | sed "s?EXTRACONDTEMPLATE?${applycorrs}?g" | sed "s?FILESOURCETEMPLATE?${filesource}?g" | sed "s?USEFILELISTTEMPLATE?False?g" | sed "s?DATASETTEMPLATE?${datasetpath}?g" | sed "s?MAXEVENTSTEMPLATE?${maxevents}?g" | sed "s?GLOBALTAGTEMPLATE?${globaltag}?g"  | sed "s?ALLFROMGTTEMPLATE?${allFromGT}?g" | sed "s?ALIGNOBJTEMPLATE?${alignobj}?g" | sed "s?GEOMTAGTEMPLATE?${taggeom}?g" | sed "s?APEOBJTEMPLATE?${apeobj}?g" | sed "s?ERRORTAGTEMPLATE?${tagape}?g" | sed "s?TRACKTYPETEMPLATE?${tracktype}?g" | sed "s?OUTFILETEMPLATE?${outfile}?g" | sed "s?VERTEXTYPETEMPLATE?${vertextype}?g" | sed "s?LUMILISTTEMPLATE?${lumilist}?g" | sed "s?PTCUTTEMPLATE?${ptcut}?g" >! ${jobname}_cfg
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
    cp ${jobname}_cfg.py  ${CMSSW_DIR}/submittedCfg/
    cp ${jobname}.lsf     ${CMSSW_DIR}/submittedCfg/
else
    echo "${CMSSW_DIR}/submittedCfg already exists"
    cp ${jobname}_cfg.py  ${CMSSW_DIR}/submittedCfg/
    cp ${jobname}.lsf     ${CMSSW_DIR}/submittedCfg/
endif

if(${options} != "--dryRun") then
 echo "cmsRun ${jobname}_cfg.py"
 cmsRun ${jobname}_cfg.py >& ${jobname}.out;

 echo "Content of working directory is: "
 \ls -lrt

 set reply=`eos find -d /store/caf/user/$USER/Alignment/PVValidation/${taskname}`
 set word=`echo $reply |awk '{split($0,a," "); print a[1]}'`

 if(${word} == "") then
 echo "Creating folder $taskname"
    eos mkdir /store/caf/user/$USER/Alignment/PVValidation/${taskname} 
 else 
    echo "Sorry /store/caf/user/$USER/Alignment/PVValidation/${taskname} already exists!"
 endif

 if (! -d  ${CMSSW_DIR}/test/PVValResults) then
     mkdir ${CMSSW_DIR}/PVValResults
     eos cp -f ${outfile} /store/caf/user/$USER/Alignment/PVValidation/${taskname}
     cp ${jobname}.out ${CMSSW_DIR}/PVValResults 
 else     
     eos cp -f ${outfile} /store/caf/user/$USER/Alignment/PVValidation/${taskname}
     cp ${jobname}.out ${CMSSW_DIR}/PVValResults 
 endif
endif

echo  "Job ended at "
\date;

exit 0
