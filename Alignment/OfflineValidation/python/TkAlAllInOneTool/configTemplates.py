from alternateValidationTemplates import *
from offlineValidationTemplates import *
from primaryVertexValidationTemplates import *
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
loadGlobalTagTemplate="""
#Global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,".oO[GlobalTag]Oo.")
"""


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
export X509_USER_PROXY=.oO[scriptsdir]Oo./.user_proxy
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`
#rfmkdir -p .oO[datadir]Oo. &>! /dev/null

#remove possible result file from previous runs
previous_results=$(eos ls /store/caf/user/$USER/.oO[eosdir]Oo.)
for file in ${previous_results}
do
    if [ ${file} = /store/caf/user/$USER/.oO[eosdir]Oo./.oO[outputFile]Oo. ]
    then
        xrdcp -f root://eoscms//eos/cms${file} root://eoscms//eos/cms${file}.bak
    fi
done

if [[ $HOSTNAME = lxplus[0-9]*[.a-z0-9]* ]] # check for interactive mode
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
eos mkdir -p /store/caf/user/$USER/.oO[eosdir]Oo.
if [ .oO[parallelJobs]Oo. -eq 1 ]
then
    root_files=$(ls --color=never -d *.oO[alignmentName]Oo.*.root)
else
    root_files=$(ls --color=never -d *.oO[alignmentName]Oo._.oO[nIndex]Oo.*.root)
fi
echo ${root_files}

for file in ${root_files}
do
    xrdcp -f ${file} root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo.
    echo ${file}
done

#cleanup
if [[ $HOSTNAME = lxplus[0-9]*[.a-z0-9]* ]] # check for interactive mode
then
    rm -rf .oO[workdir]Oo.
fi
echo "done."
"""


######################################################################
######################################################################
cfgTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process(".oO[ProcessName]Oo.")

.oO[datasetDefinition]Oo.
.oO[Bookkeeping]Oo.
.oO[LoadBasicModules]Oo.
.oO[TrackSelectionRefitting]Oo.
.oO[LoadGlobalTagTemplate]Oo.
.oO[condLoad]Oo.
.oO[ValidationConfig]Oo.
.oO[FileOutputTemplate]Oo.

.oO[DefinePath]Oo.
"""


######################################################################
######################################################################
Bookkeeping = """
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.statistics.append('cout')
"""


######################################################################
######################################################################
CommonTrackSelectionRefitting = """
import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(process, '.oO[trackcollection]Oo.',
                                                     TTRHBuilder='.oO[ttrhbuilder]Oo.',
                                                     usePixelQualityFlag=.oO[usepixelqualityflag]Oo.,
                                                     openMassWindow=.oO[openmasswindow]Oo.,
                                                     cosmicsDecoMode=.oO[cosmicsdecomode]Oo.,
                                                     cosmicsZeroTesla=.oO[cosmics0T]Oo.,
                                                     momentumConstraint=.oO[momentumconstraint]Oo.,
                                                     cosmicTrackSplitting=.oO[istracksplitting]Oo.,
                                                     use_d0cut=.oO[use_d0cut]Oo.,
                                                    )

.oO[trackhitfiltercommands]Oo.
"""


######################################################################
######################################################################
SingleTrackRefitter = """
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = ".oO[TrackCollection]Oo."
process.TrackRefitter.TTRHBuilder = ".oO[ttrhbuilder]Oo."
process.TrackRefitter.NavigationSchool = ""
"""


######################################################################
######################################################################
LoadBasicModules = """
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.Geometry.GeometryDB_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences..oO[magneticField]Oo._cff")
"""


######################################################################
######################################################################
FileOutputTemplate = """
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('.oO[outputFile]Oo.')
)
"""


######################################################################
######################################################################
DefinePath_CommonSelectionRefitting = """
process.p = cms.Path(
process.seqTrackselRefit*.oO[ValidationSequence]Oo.)
"""

######################################################################
######################################################################
mergeTemplate="""
#!/bin/bash
CWD=`pwd -P`
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`


.oO[createResultsDirectory]Oo.

if [[ $HOSTNAME = lxplus[0-9]*[.a-z0-9]* ]] # check for interactive mode
then
    mkdir -p .oO[workdir]Oo.
    cd .oO[workdir]Oo.
else
    cd $CWD
fi
echo "Working directory: $(pwd -P)"

###############################################################################
# download root files from eos
root_files=$(eos ls /store/caf/user/$USER/.oO[eosdir]Oo. \
             | grep ".root$" | grep -v "result.root$")
#for file in ${root_files}
#do
#    xrdcp -f root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./${file} .
#    echo ${file}
#done


#run
.oO[DownloadData]Oo.
.oO[CompareAlignments]Oo.

.oO[RunValidationPlots]Oo.

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
mergeParallelOfflineTemplate="""
#!/bin/bash
eos='/afs/cern.ch/project/eos/installation/cms/bin/eos.select'
CWD=`pwd -P`
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`

if [[ $HOSTNAME = lxplus[0-9]*[.a-z0-9]* ]] # check for interactive mode
then
    mkdir -p .oO[workdir]Oo.
    cd .oO[workdir]Oo.
else
    cd $CWD
fi
echo "Working directory: $(pwd -P)"

###############################################################################
# download root files from eos
root_files=$(ls /eos/cms/store/caf/user/$USER/.oO[eosdir]Oo. \
             | grep ".root$" | grep -v "result.root$")
#for file in ${root_files}
#do
#    xrdcp -f root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./${file} .
#    echo ${file}
#done


#run
.oO[DownloadData]Oo.
"""

######################################################################
######################################################################
createResultsDirectoryTemplate="""
#create results-directory and copy used configuration there
rfmkdir -p .oO[datadir]Oo.
rfcp .oO[logdir]Oo./usedConfiguration.ini .oO[datadir]Oo.
"""


######################################################################
######################################################################
mergeParallelResults="""

.oO[beforeMerge]Oo.
.oO[doMerge]Oo.

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
eos mkdir -p /store/caf/user/$USER/.oO[eosdir]Oo./
cp .oO[Alignment/OfflineValidation]Oo./scripts/compareFileAges.C .
root -x -q -b -l "compareFileAges.C(\\\"root://eoscms.cern.ch//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./.oO[validationId]Oo._result.root\\\", \\\".oO[compareStringsPlain]Oo.\\\")"
comparisonNeeded=${?}

if [[ ${comparisonNeeded} -eq 1 ]]
then
    cp .oO[compareAlignmentsPath]Oo. .
    root -x -q -b -l '.oO[compareAlignmentsName]Oo.++(\".oO[compareStrings]Oo.\", ".oO[legendheader]Oo.", ".oO[customtitle]Oo.", ".oO[customrighttitle]Oo.", .oO[bigtext]Oo.)'
    mv result.root .oO[validationId]Oo._result.root
    xrdcp -f .oO[validationId]Oo._result.root root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo.
else
    echo ".oO[validationId]Oo._result.root is up-to-date, no need to compare again."
    xrdcp -f root://eoscms//eos/cms/store/caf/user/$USER/.oO[eosdir]Oo./.oO[validationId]Oo._result.root .
fi
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
