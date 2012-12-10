from alternateValidationTemplates import *
from offlineValidationTemplates import *
from geometryComparisonTemplates import *
from monteCarloValidationTemplates import *
from trackSplittingValidationTemplates import *
from zMuMuValidationTemplates import *


######################################################################
######################################################################
###                                                                ###
###                       General Templates                        ###
###                                                                ###
######################################################################
######################################################################

######################################################################
######################################################################
dbLoadTemplate="""
##include private db object
##
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.trackerAlignment =  CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
                                        connect = cms.string('.oO[dbpath]Oo.'),
#                                         timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                   tag = cms.string('.oO[tag]Oo.')
                                                                   ))
                                        )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

"""


######################################################################
######################################################################
APETemplate="""
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.APE = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
                                        connect = cms.string('.oO[errordbpath]Oo.'),
#                                         timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                                   tag = cms.string('.oO[errortag]Oo.')
                                                                   ))
                                        )
process.es_prefer_APE = cms.ESPrefer("PoolDBESSource", "APE")
"""


######################################################################
######################################################################
kinksAndBowsTemplate="""
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.trackerBowedSensors = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
     connect = cms.string('.oO[kbdbpath]Oo.'),
     toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerSurfaceDeformationRcd'),
                               tag = cms.string('.oO[kbtag]Oo.')
                               )
                      )
    )
process.prefer_trackerBowedSensors = cms.ESPrefer("PoolDBESSource", "trackerBowedSensors")
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
source /afs/cern.ch/cms/caf/setup.sh
# source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
# export SCRAM_ARCH=slc5_amd64_gcc462
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`
rfmkdir -p .oO[workdir]Oo.
rfmkdir -p .oO[datadir]Oo.

rm -f .oO[workdir]Oo./*
cd .oO[workdir]Oo.

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
rfmkdir -p .oO[logdir]Oo.
gzip -f LOGFILE_*_.oO[name]Oo..log
find .oO[workdir]Oo. -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."
rfmkdir -p .oO[datadir]Oo.
find .oO[workdir]Oo. -maxdepth 1 -name "*.oO[alignmentName]Oo.*.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."
#cleanup
rm -rf .oO[workdir]Oo.
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
rfmkdir -p .oO[datadir]Oo.

#remove possible result file from previous runs
if [ -e .oO[datadir]Oo./*.oO[alignmentName]Oo..root ] ; then
    for file in .oO[datadir]Oo./*.oO[alignmentName]Oo..root; do mv -f $file $file.bak; done
    #rm -f  .oO[datadir]Oo./*.oO[alignmentName]Oo..root
fi    

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
rfmkdir -p .oO[logdir]Oo.
gzip LOGFILE_*_.oO[name]Oo..log
find ${LSFWORKDIR} -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."
rfmkdir -p .oO[datadir]Oo.
find ${LSFWORKDIR} -maxdepth 1 -name "*.oO[alignmentName]Oo._.oO[nIndex]Oo.*.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."
#cleanup - do not remove workdir, since another parallel job might be running in the same node
find ${LSFWORKDIR} -maxdepth 1 -name "*.oO[alignmentName]Oo._.oO[nIndex]Oo.*.root" -print | xargs -I {} bash -c "rm {}"
echo "done."
"""


######################################################################
######################################################################
mergeTemplate="""
#!/bin/bash
#init
export STAGE_SVCCLASS=cmscafuser
# source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
# export SCRAM_ARCH=slc5_amd64_gcc462
export SCRAM_ARCH=.oO[SCRAM_ARCH]Oo.
eval `scramv1 ru -sh`
rfmkdir -p .oO[workdir]Oo.
cd .oO[workdir]Oo.

#run
.oO[DownloadData]Oo.
.oO[CompareAllignments]Oo.

find ./ -maxdepth 1 -name "*_result.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."

.oO[RunExtendedOfflineValidation]Oo.

#zip stdout and stderr from the farm jobs
cd .oO[logdir]Oo.
find . -name "*.stderr" -exec gzip -f {} \;
find . -name "*.stdout" -exec gzip -f {} \;
"""


######################################################################
######################################################################
compareAlignmentsExecution="""
#merge for .oO[validationId]Oo.
root -q -b '.oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/compareAlignments.cc+(\".oO[compareStrings]Oo.\")'
mv result.root .oO[validationId]Oo._result.root
"""


######################################################################
######################################################################
extendedValidationExecution="""
#run extended offline validation scripts
rfmkdir -p .oO[workdir]Oo./ExtendedOfflineValidation_Images
root -x -b -q .oO[extendeValScriptPath]Oo.
rfmkdir -p .oO[datadir]Oo./ExtendedOfflineValidation_Images
find .oO[workdir]Oo./ExtendedOfflineValidation_Images -maxdepth 1 -name \"*ps\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./ExtendedOfflineValidation_Images\"
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

  .oO[extendedInstantiation]Oo.
  gROOT->ProcessLine(".mkdir -p .oO[workdir]Oo./ExtendedOfflineValidation_Images/");
  p.setOutputDir(".oO[workdir]Oo./ExtendedOfflineValidation_Images");
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
datasetpath = .oO[datasetCrab]Oo.
pset = .oO[cfgFile]Oo.
## for MC
# total_number_of_events = .oO[nEvents]Oo.
# events_per_job = 5000
## for real Data
##Data
total_number_of_lumis = -1
number_of_jobs = .oO[numberOfJobs]Oo.
output_file = .oO[outputFile]Oo.
runselection = .oO[runRange]Oo.
lumi_mask = .oO[JSON]Oo.

[USER]
return_data = 0
copy_data = 1
storage_element = T2_CH_CERN
user_remote_dir	= AlignmentValidation/.oO[crabOutputDir]Oo.
ui_working_dir = .oO[crabWorkingDir]Oo.
# script_exe = .oO[script]Oo.

[CAF]
queue = .oO[queue]Oo.
"""


######################################################################
######################################################################
crabShellScriptTemplate="""
cd .oO[crabBaseDir]Oo.

# source the needed environment for crab in the right order
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env..oO[useCshell]Oo.sh
cmsenv
source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab..oO[useCshell]Oo.sh

# Create and submit parallel jobs
.oO[crabCommand]Oo.

cd -
"""


######################################################################
######################################################################
crabCommandTemplate="""
crab -create -cfg .oO[crabCfgName]Oo.
crab -submit -c .oO[crabWorkingDir]Oo.
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
        raise StandardError, "unkown template to replace %s"%templateName
    if not alternateTemplateName in globals().keys():
        raise StandardError, "unkown template to replace %s"%alternateTemplateName
    globals()[ templateName ] = globals()[ alternateTemplateName ]
    # = eval("configTemplates.%s"%"alternateTemplate")
