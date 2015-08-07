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
#init
#ulimit -v 3072000
#export STAGE_SVCCLASS=cmscafuser
#save path to the LSF batch working directory  (/pool/lsf)
export LSFWORKDIR=`pwd -P`
echo LSF working directory is $LSFWORKDIR
source /afs/cern.ch/cms/caf/setup.sh
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`
#rfmkdir -p .oO[datadir]Oo. &>! /dev/null

#remove possible result file from previous runs
previous_results=$(cmsLs -l /store/caf/user/$USER/.oO[eosdir]Oo. | awk '{print $5}')
for file in ${previous_results}
do
    if [ ${file} = /store/caf/user/$USER/.oO[eosdir]Oo./.oO[outputFile]Oo. ]
    then
        cmsStage -f ${file} ${file}.bak
    fi
done

if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    rfmkdir -p .oO[workdir]Oo.
    rm -f .oO[workdir]Oo./*
    cd .oO[workdir]Oo.
else
    mkdir -p $LSFWORKDIR/TkAllInOneTool
    cd $LSFWORKDIR/TkAllInOneTool
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
find . -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."

#copy root files to eos
cmsMkdir /store/caf/user/$USER/.oO[eosdir]Oo.
if [ .oO[parallelJobs]Oo. -eq 1 ]
then
    root_files=$(ls --color=never -d *.oO[alignmentName]Oo.*.root)
else
    root_files=$(ls --color=never -d *.oO[alignmentName]Oo._.oO[nIndex]Oo.*.root)
fi
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
mergeTemplate="""
#!/bin/bash
CWD=`pwd -P`
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`

#create results-directory and copy used configuration there
rfmkdir -p .oO[datadir]Oo.
rfcp .oO[logdir]Oo./usedConfiguration.ini .oO[datadir]Oo.

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
.oO[RunTrackSplitPlot]Oo.

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
mergeParallelResults="""

.oO[copyMergeScripts]Oo.
.oO[haddLoop]Oo.

# create log file
ls -al .oO[mergeParallelFilePrefixes]Oo. > .oO[datadir]Oo./log_rootfilelist.txt

# Remove parallel job files
.oO[rmUnmerged]Oo.
"""


######################################################################
######################################################################
compareAlignmentsExecution="""
#merge for .oO[validationId]Oo. if it does not exist or is not up-to-date
echo -e "\n\nComparing validations"
cmsMkdir /store/caf/user/$USER/.oO[eosdir]Oo./
cp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/compareFileAges.C .
root -x -q -b -l "compareFileAges.C(\\\"root://eoscms.cern.ch//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./.oO[validationId]Oo._result.root\\\", \\\".oO[compareStringsPlain]Oo.\\\")"
comparisonNeeded=${?}

if [[ ${comparisonNeeded} -eq 1 ]]
then
    cp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/compareAlignments.cc .
    root -x -q -b -l 'compareAlignments.cc++(\".oO[compareStrings]Oo.\")'
    mv result.root .oO[validationId]Oo._result.root
    cmsStage -f .oO[validationId]Oo._result.root /store/caf/user/$USER/.oO[eosdir]Oo.
else
    echo ".oO[validationId]Oo._result.root is up-to-date, no need to compare again."
    cmsStage -f /store/caf/user/$USER/.oO[eosdir]Oo./.oO[validationId]Oo._result.root .
fi
"""


######################################################################
######################################################################
extendedValidationExecution="""
#run extended offline validation scripts
echo -e "\n\nRunning extended offline validation"
if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    rfmkdir -p .oO[workdir]Oo./ExtendedOfflineValidation_Images
else
    mkdir -p ExtendedOfflineValidation_Images
fi

rfcp .oO[extendedValScriptPath]Oo. .
rfcp .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/macros/PlotAlignmentValidation.C .
root -x -b -q -l TkAlExtendedOfflineValidation.C
rfmkdir -p .oO[datadir]Oo./ExtendedOfflineValidation_Images

if [[ $HOSTNAME = lxplus[0-9]*\.cern\.ch ]] # check for interactive mode
then
    image_files=$(ls --color=never | find .oO[workdir]Oo./ExtendedOfflineValidation_Images/ -name \*ps -o -name \*root)
    echo -e "\n\nProduced plot files:"
    #echo ${image_files}
    ls .oO[workdir]Oo./ExtendedOfflineValidation_Images
else
    image_files=$(ls --color=never | find ExtendedOfflineValidation_Images/ -name \*ps -o -name \*root)
    echo -e "\n\nProduced plot files:"
    #echo ${image_files}
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
#include ".oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/macros/PlotAlignmentValidation.C"
void TkAlExtendedOfflineValidation()
{
  // load framework lite just to find the CMSSW libs...
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  //compile the makro
  //gROOT->ProcessLine(".L .oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/macros/PlotAlignmentValidation.C++");
  // gROOT->ProcessLine(".L ./PlotAlignmentValidation.C++");

  .oO[extendedInstantiation]Oo.
  p.setOutputDir("./ExtendedOfflineValidation_Images");
  p.setTreeBaseDir(".oO[OfflineTreeBaseDir]Oo.");
  p.plotDMR(".oO[DMRMethod]Oo.",.oO[DMRMinimum]Oo.,".oO[DMROptions]Oo.");
  p.plotSurfaceShapes(".oO[SurfaceShapes]Oo.");
  p.plotChi2("root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./.oO[resultPlotFile]Oo._result.root");
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
        msg = "unknown template to replace %s"%templateName
        raise AllInOneError(msg)
    if not alternateTemplateName in globals().keys():
        msg = "unknown template to replace %s"%alternateTemplateName
        raise AllInOneError(msg)
    globals()[ templateName ] = globals()[ alternateTemplateName ]
    # = eval("configTemplates.%s"%"alternateTemplate")
