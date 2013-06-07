from alternateValidationTemplates import *
from offlineValidationTemplates import *
from geometryComparisonTemplates import *
from monteCarloValidationTemplates import *
from trackSplittingValidationTemplates import *
from zMuMuValidationTemplates import *
from TkAlExceptions import AllInOneError


######################################################################
######################################################################
###                                                                ###
###                       General Templates                        ###
###                                                                ###
######################################################################
######################################################################

######################################################################
######################################################################
conditionsTemplate="""
process.conditionsIn.oO[rcdName]Oo. = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
     connect = cms.string('.oO[connectString]Oo.'),
     toGet = cms.VPSet(cms.PSet(record = cms.string('.oO[rcdName]Oo.'),
                               tag = cms.string('.oO[tagName]Oo.')
                               )
                      )
    )
process.prefer_conditionsIn.oO[rcdName]Oo. = cms.ESPrefer("PoolDBESSource", "conditionsIn.oO[rcdName]Oo.")
"""


######################################################################
######################################################################
#batch job execution
scriptTemplate="""
#!/bin/bash
CWD=`pwd -P`
source /afs/cern.ch/cms/caf/setup.sh
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`
# rfmkdir -p .oO[workdir]Oo.
# rfmkdir -p .oO[datadir]Oo.

if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    rfmkdir -p .oO[workdir]Oo.
    rm -f .oO[workdir]Oo./*
    cd .oO[workdir]Oo.
else
    mkdir -p $CWD/TkAllInOneTool
    cd $CWD/TkAllInOneTool
fi

# rm -f .oO[workdir]Oo./*
# cd .oO[workdir]Oo.

#run
pwd
df -h .
.oO[CommandLine]Oo.
echo "----"
echo "List of files in $(pwd):"
ls -ltr
echo "----"
echo ""


#retrieve
rfmkdir -p .oO[logdir]Oo. >&! /dev/null
gzip -f LOGFILE_*_.oO[name]Oo..log
find .oO[workdir]Oo. -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."

#copy root files to eos
cmsMkdir /store/caf/user/$USER/.oO[eosdir]Oo.
root_files=$(ls --color=never -d *.oO[alignmentName]Oo.*.root)
echo ${root_files}

for file in ${root_files}
do
    cmsStage -f ${file} /store/caf/user/$USER/.oO[eosdir]Oo.
    echo ${file}
done

#cleanup
if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    rm -rf .oO[workdir]Oo.
fi
echo "done."
"""


######################################################################
######################################################################
#batch job execution
parallelScriptTemplate="""
#!/bin/bash
#init
#ulimit -v 3072000
#export STAGE_SVCCLASS=cmscafuser
#save path to the LSF batch working directory  (/pool/lsf)
export LSFWORKDIR=$PWD
echo LSF working directory is $LSFWORKDIR
source /afs/cern.ch/cms/caf/setup.sh
# source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
# export SCRAM_ARCH=slc5_amd64_gcc462
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`
#rfmkdir -p ${LSFWORKDIR}

# make rfmkdir silent in case directory already exists
rfmkdir -p .oO[datadir]Oo. >&! /dev/null
cmsMkdir /store/caf/user/$USER/.oO[eosdir]Oo.

#remove possible result file from previous runs
previous_results=$(cmsLs -l /store/caf/user/$USER/.oO[eosdir]Oo. | awk '{print $5}')
for file in ${previous_results}
do
    # if [ ${file} = *.oO[datadir]Oo./*.oO[alignmentName]Oo.*.root ]
    if [ ${file} = /store/caf/user/$USER/.oO[eosdir]Oo./.oO[outputFile]Oo. ]
    then
        cmsStage -f ${file} ${file}.bak
    fi
done

#rm -f ${LSFWORKDIR}/*
cd ${LSFWORKDIR}

#run
pwd
df -h .
.oO[CommandLine]Oo.
echo "----"
echo "List of files in $(pwd):"
ls -ltr
echo "----"
echo ""


#retrieve
rfmkdir -p .oO[logdir]Oo. >&! /dev/null
gzip LOGFILE_*_.oO[name]Oo..log
find ${LSFWORKDIR} -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."

#copy root files to eos
cmsMkdir /store/caf/user/$USER/.oO[eosdir]Oo.
root_files=$(ls --color=never -d ${LSFWORKDIR}/*.oO[alignmentName]Oo._.oO[nIndex]Oo.*.root)
echo "ls"
ls
echo "\${root_files}:"
echo ${root_files}
for file in ${root_files}
do
    # echo "cmsStage -f ${file} /store/caf/user/$USER/.oO[eosdir]Oo."
    cmsStage -f ${file} /store/caf/user/$USER/.oO[eosdir]Oo.
    # echo ${file}
done

#cleanup - do not remove workdir, since another parallel job might be running in the same node
find ${LSFWORKDIR} -maxdepth 1 -name "*.oO[alignmentName]Oo._.oO[nIndex]Oo.*.root" -print | xargs -I {} bash -c "rm {}"
echo "done."
"""


######################################################################
######################################################################
mergeTemplate="""
#!/bin/bash
CWD=`pwd -P`
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`

if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    mkdir -p .oO[workdir]Oo.
    cd .oO[workdir]Oo.
else
    cd $CWD
fi
echo "Working directory: $(pwd -P)"

###############################################################################
# download root files from eos
root_files=$(cmsLs -l /store/caf/user/$USER/.oO[eosdir]Oo. | awk '{print $5}' \
             | grep ".root$" | grep -v "result.root$")
for file in ${root_files}
do
    cmsStage -f ${file} .
    # echo ${file}
done


#run
.oO[DownloadData]Oo.
.oO[CompareAlignments]Oo.

.oO[RunExtendedOfflineValidation]Oo.

for file in $(ls -d --color=never *_result.root)
do
    cmsStage -f ${file} /store/caf/user/$USER/.oO[eosdir]Oo.
done

# clean-up
# ls -l *.root
rm -f *.root

#zip stdout and stderr from the farm jobs
cd .oO[logdir]Oo.
find . -name "*.stderr" -exec gzip -f {} \;
find . -name "*.stdout" -exec gzip -f {} \;
"""


######################################################################
######################################################################
compareAlignmentsExecution="""
#merge for .oO[validationId]Oo.
cp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/compareAlignments.cc .
root -q -b 'compareAlignments.cc++(\".oO[compareStrings]Oo.\")'
mv result.root .oO[validationId]Oo._result.root
"""


######################################################################
######################################################################
extendedValidationExecution="""
#run extended offline validation scripts
if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    rfmkdir -p .oO[workdir]Oo./ExtendedOfflineValidation_Images
else
    mkdir -p ExtendedOfflineValidation_Images
fi

rfcp .oO[extendeValScriptPath]Oo. .
rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/macros/PlotAlignmentValidation.C .
root -x -b -q TkAlExtendedOfflineValidation.C
rfmkdir -p .oO[datadir]Oo./ExtendedOfflineValidation_Images

if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    image_files=$(ls --color=never .oO[workdir]Oo./ExtendedOfflineValidation_Images/*ps)
    echo ${image_files}
    ls .oO[workdir]Oo./ExtendedOfflineValidation_Images
else
    image_files=$(ls --color=never ExtendedOfflineValidation_Images/*ps)
    echo ${image_files}
    ls ExtendedOfflineValidation_Images
fi

for image in ${image_files}
do
    cp ${image} .oO[datadir]Oo./ExtendedOfflineValidation_Images
done
"""


######################################################################
######################################################################
extendedValidationTemplate="""
void TkAlExtendedOfflineValidation()
{
  // load framework lite just to find the CMSSW libs...
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  //compile the makro
  gROOT->ProcessLine(".L .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/macros/PlotAlignmentValidation.C++");
  // gROOT->ProcessLine(".L ./PlotAlignmentValidation.C++");

  .oO[extendedInstantiation]Oo.
  p.setOutputDir("./ExtendedOfflineValidation_Images");
  p.setTreeBaseDir(".oO[OfflineTreeBaseDir]Oo.");
  p.plotDMR(".oO[DMRMethod]Oo.",.oO[DMRMinimum]Oo.,".oO[DMROptions]Oo.");
  p.plotSurfaceShapes(".oO[SurfaceShapes]Oo.");
}
"""


######################################################################
######################################################################
crabCfgTemplate="""
[CRAB]
jobtype = cmssw
scheduler = caf
use_server = 0

[CMSSW]
datasetpath = .oO[dataset]Oo.
pset = .oO[cfgFile]Oo.
total_number_of_.oO[McOrData]Oo.
number_of_jobs = .oO[numberOfJobs]Oo.
output_file = .oO[outputFile]Oo.
runselection = .oO[runRange]Oo.
lumi_mask = .oO[JSON]Oo.

[USER]
return_data = 0
copy_data = 1
storage_element = T2_CH_CERN
user_remote_dir	= .oO[eosdir]Oo.
ui_working_dir = .oO[crabWorkingDir]Oo.
# script_exe = .oO[script]Oo.
# .oO[email]Oo.

[CAF]
queue = .oO[queue]Oo.
"""




######################################################################
######################################################################
###                                                                ###
###                      Alternate Templates                       ###
###                                                                ###
######################################################################
######################################################################


def alternateTemplate( templateName, alternateTemplateName ):
  
    if not templateName in globals().keys():
        msg = "unkown template to replace %s"%templateName
        raise AllInOneError(msg) 
    if not alternateTemplateName in globals().keys():
        msg = "unkown template to replace %s"%alternateTemplateName
        raise AllInOneError(msg) 
    globals()[ templateName ] = globals()[ alternateTemplateName ]
    # = eval("configTemplates.%s"%"alternateTemplate")
