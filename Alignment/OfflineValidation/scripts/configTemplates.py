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
#from CondCore.DBCommon.CondDBSetup_cfi import *
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
##include private db object
##
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.trackerAlignment =  CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
                                        connect = cms.string('.oO[dbpath]Oo.'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                   tag = cms.string('.oO[tag]Oo.')
                                                                   ))
                                        )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

"""


######################################################################
######################################################################
APETemplate="""
from CondCore.DBCommon.CondDBSetup_cfi import *
process.APE = poolDBESSource.clone(
                                        connect = cms.string('.oO[errordbpath]Oo.'),
                                        timetype = cms.string("runnumber"),
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
#batch job execution
scriptTemplate="""
#!/bin/bash
#init
#ulimit -v 3072000
#export STAGE_SVCCLASS=cmscafuser
source /afs/cern.ch/cms/caf/setup.sh
# source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
export SCRAM_ARCH=slc5_amd64_gcc462
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
gzip LOGFILE_*_.oO[name]Oo..log
find .oO[workdir]Oo. -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."
rfmkdir -p .oO[datadir]Oo.
find .oO[workdir]Oo. -maxdepth 1 -name "*.oO[alignmentName]Oo.*.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."
#cleanup
rm -rf .oO[workdir]Oo.
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
export SCRAM_ARCH=slc5_amd64_gcc462
eval `scramv1 ru -sh`
rfmkdir -p .oO[workdir]Oo.
cd .oO[workdir]Oo.

#run
.oO[DownloadData]Oo.
.oO[CompareAllignments]Oo.

find ./ -maxdepth 1 -name "*_result.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."

.oO[RunExtendedOfflineValidation]Oo.

#zip stdout and stderr from the farm jobs
gzip .oO[logdir]Oo./*.stderr
gzip .oO[logdir]Oo./*.stdout

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
  p.plotDMR(".oO[DMRMethod]Oo.",.oO[DMRMinimum]Oo.);
}
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
