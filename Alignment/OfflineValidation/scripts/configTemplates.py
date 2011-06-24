###
#                 Default Templates
###

offlineTemplate = """
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
reportEvery = cms.untracked.int32(1000) # every 1000th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
))
process.MessageLogger.statistics.append('cout')

#############################################################
##select trigger bits 40 OR 41
##AND NOT (36 OR 37 OR 38 OR 39)
##trigger bit 0 is selecting crossing bunches (not in MC)
##########################################################
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

#Good Bunch Crossings
process.bptxAnd = process.hltLevel1GTSeed.clone(L1TechTriggerSeeding = cms.bool(True), L1SeedsLogicalExpression = cms.string('0'))
#BSCNOBEAMHALO
process.bit40 = process.hltLevel1GTSeed.clone(L1TechTriggerSeeding = cms.bool(True), L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))'))

#Physics declared
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

    
#############################################################
##select only high purity tracks
##has to run first as necessary information
##is only available in initial track selection
##(Quality information is thrown away by the tracker refitters)
##########################################################
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
process.HighPuritySelector = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    applyBasicCuts = True,
    filter = True,
    src = '.oO[TrackCollection]Oo.',
    trackQualities = ["highPurity"]
    )

###############################################################
## Quality filters on the event (REAL DATA - FIRST COLLISIONS DEC09 ONLY!)
## see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TRKPromptFeedBack#Event_and_track_selection_recipe
##NOT in PATH yet, to be commented in, if necessay
##
#########################################################    
############################################################################
##Produce Primary Vertex Collection needed for later analysis
############################################################################
#process.load('TrackingTools/TransientTrack/TransientTrackBuilder_cfi')
#process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
#process.offlinePrimaryVertices.TrackLabel = 'ALCARECOTkAlMinBias'

process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag("offlinePrimaryVertices"),
                                           cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
                                           filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
                                           )



process.FilterGoodEvents=cms.Sequence(#process.HighPuritySelector*
process.oneGoodVertexFilter)


process.noScraping= cms.EDFilter("FilterOutScraping",
                                 src=cms.InputTag("ALCARECOTkAlMinBias"),
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.25)
                                 )
####################################





#-- Track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'

#-- Alignment Track Selection
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'HitFilteredTracks'
process.AlignmentTrackSelector.filter = True

.oO[TrackSelectionTemplate]Oo.

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
   src = 'TrackerTrackHitFilter',
   NavigationSchool = "",

###    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate"    
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

#############
# parameters for TrackRefitter
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = process.TrackRefitter.clone(
   src = 'HighPuritySelector',
   TrajectoryInEvent = True,
   TTRHBuilder = "WithAngleAndTemplate",
   NavigationSchool = ""
)
process.TrackRefitter2 = process.TrackRefitter1.clone(
#    src = 'HitFilteredTracks')
     src = 'AlignmentTrackSelector'
)


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."


.oO[LorentzAngleTemplate]Oo.
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
process.load("Configuration/StandardSequences/MagneticField_38T_cff")

.oO[dbLoad]Oo.

.oO[APE]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation and Output File
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = .oO[offlineModuleLevelHistsTransient]Oo.
.oO[offlineValidationFileOutput]Oo.

 ##
 ## PATH
 ##
process.p = cms.Path(
process.triggerSelection*
process.offlineBeamSpot*process.HighPuritySelector*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
*process.AlignmentTrackSelector*process.TrackRefitter2*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)

"""

offlineStandaloneFileOutputTemplate = """
process.TFileService.fileName = '.oO[outputFile]Oo.'
"""

offlineDqmFileOutputTemplate = """
process.TrackerOfflineValidationSummary.oO[offlineValidationMode]Oo..removeModuleLevelHists = .oO[offlineModuleLevelHistsTransient]Oo.
process.DqmSaverTkAl.workflow = '.oO[workflow]Oo.'
process.DqmSaverTkAl.dirName = '.oO[workdir]Oo./.'
process.DqmSaverTkAl.forceRunNumber = .oO[firstRunNumber]Oo.
"""
#offlineDqmFileOutputTemplate = """
#process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..workflow =  .oO[workflow]Oo.
#process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..dirName = .oO[workdir]Oo./.
#process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..forceRunNumber = .oO[firstRunNumber]Oo.
#"""


LorentzAngleTemplate = "#use lorentz angle from global tag"

TrackSelectionTemplate = """
#####default for MC tracks with now further corrections etc.

process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 3
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.65
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.17 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

##############
##Trigger sequence
#############
#for MC only trigger bit 40 is simulated
#no triger on bunch crossing bit 0


process.triggerSelection=cms.Sequence(process.bit40)

"""

intoNTuplesTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("ValidationIntoNTuples")

# global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo." 

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
) 

#removed: APE
#removed: dbLoad
import CalibTracker.Configuration.Common.PoolDBESSource_cfi
process.GeomToComp = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
connect = cms.string('.oO[dbpath]Oo.'),

    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('.oO[tag]Oo.')
    ))
   
)
process.es_prefer_geom=cms.ESPrefer("PoolDBESSource","GeomToComp")

from CondCore.DBCommon.CondDBSetup_cfi import *

process.ZeroAPE = cms.ESSource("PoolDBESSource",CondDBSetup,
								connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
								timetype = cms.string("runnumber"),
								toGet = cms.VPSet(
											cms.PSet(
												record = cms.string('TrackerAlignmentErrorRcd'),
												tag = cms.string('TrackerIdealGeometryErrors210_mc')
											))
								)
process.es_prefer_ZeroAPE = cms.ESPrefer("PoolDBESSource", "ZeroAPE")


process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.dump = cms.EDAnalyzer("TrackerGeometryIntoNtuples",
    outputFile = cms.untracked.string('.oO[workdir]Oo./.oO[alignmentName]Oo.ROOTGeometry.root'),
    outputTreename = cms.untracked.string('alignTree')
)

process.p = cms.Path(process.dump)  
"""

compareTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")

# global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo." 
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

#process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Frontier_cff")
# the input .GlobalPosition_Frontier_cff is providing the frontier://FrontierProd/CMS_COND_31X_ALIGNMENT in the release which does not provide the ideal geometry
#process.GlobalPosition.connect = 'frontier://FrontierProd/CMS_COND_31X_FROM21X'

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo', 
        'cout')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

  # configuration of the Tracker Geometry Comparison Tool
  # Tracker Geometry Comparison
process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")
  # the input "IDEAL" is special indicating to use the ideal geometry of the release

process.TrackerGeometryCompare.inputROOTFile1 = '.oO[comparedGeometry]Oo.'
process.TrackerGeometryCompare.inputROOTFile2 = '.oO[referenceGeometry]Oo.'
process.TrackerGeometryCompare.outputFile = ".oO[workdir]Oo./.oO[name]Oo..Comparison_common.oO[common]Oo..root"
process.TrackerGeometryCompare.levels = [ .oO[levels]Oo. ]

  ##FIXME!!!!!!!!!
  ##replace TrackerGeometryCompare.writeToDB = .oO[dbOutput]Oo.
  ##removed: dbOutputService

process.p = cms.Path(process.TrackerGeometryCompare)
"""
  
dbOutputTemplate= """
//_________________________ db Output ____________________________
        # setup for writing out to DB
        include "CondCore/DBCommon/data/CondDBSetup.cfi"
#       include "CondCore/DBCommon/data/CondDBCommon.cfi"

    service = PoolDBOutputService {
        using CondDBSetup
        VPSet toPut = {
            { string record = "TrackerAlignmentRcd"  string tag = ".oO[tag]Oo." },
            { string record = "TrackerAlignmentErrorRcd"  string tag = ".oO[errortag]Oo." }
        }
                string connect = "sqlite_file:.oO[workdir]Oo./.oO[name]Oo.Common.oO[common]Oo..db"
                # untracked string catalog = "file:alignments.xml"
        untracked string timetype = "runnumber"
    }
"""

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



#batch job execution
scriptTemplate="""
#!/bin/bash
#init
ulimit -v 3072000
#export STAGE_SVCCLASS=cmscafuser
source /afs/cern.ch/cms/caf/setup.sh
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
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


#retrive
rfmkdir -p .oO[logdir]Oo.
gzip LOGFILE_*_.oO[name]Oo..log
find .oO[workdir]Oo. -maxdepth 1 -name "LOGFILE*.oO[alignmentName]Oo.*" -print | xargs -I {} bash -c "rfcp {} .oO[logdir]Oo."
rfmkdir -p .oO[datadir]Oo.
find .oO[workdir]Oo. -maxdepth 1 -name "*.oO[alignmentName]Oo.*.root" -print | xargs -I {} bash -c "rfcp {} .oO[datadir]Oo."
#cleanup
rm -rf .oO[workdir]Oo.
echo "done."
"""

mergeTemplate="""
#!/bin/bash
#init
export STAGE_SVCCLASS=cmscafuser
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd .oO[CMSSW_BASE]Oo./src
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

compareAlignmentsExecution="""
#merge for .oO[validationId]Oo.
root -q -b '.oO[CMSSW_BASE]Oo./src/Alignment/OfflineValidation/scripts/compareAlignments.cc+(\".oO[compareStrings]Oo.\")'
mv result.root .oO[validationId]Oo._result.root
"""

extendedVaidationExecution="""
#run extended offline validation scripts
rfmkdir -p .oO[workdir]Oo./ExtendedOfflineValidation_Images
root -x -b -q .oO[extendeValScriptPath]Oo.
rfmkdir -p .oO[datadir]Oo./ExtendedOfflineValidation_Images
find .oO[workdir]Oo./ExtendedOfflineValidation_Images -maxdepth 1 -name \"*ps\" -print | xargs -I {} bash -c \"rfcp {} .oO[datadir]Oo./ExtendedOfflineValidation_Images\"
"""

extendedVaidationTemplate="""
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

mcValidateTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("TkVal")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_McValidate_.oO[name]Oo.', 
        'cout')
)

### standard includes
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### conditions
process.load("Alignment.OfflineValidation.GlobalTag_cff")
process.GlobalTag.globaltag = '.oO[GlobalTag]Oo.'

import CalibTracker.Configuration.Common.PoolDBESSource_cfi

.oO[dbLoad]Oo.

### validation-specific includes
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

### configuration MultiTrackValidator ###
process.multiTrackValidator.outputFile = '.oO[outputFile]Oo.'

process.multiTrackValidator.associators = ['TrackAssociatorByHits']
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.label = ['generalTracks']

from Alignment.OfflineValidation..oO[RelValSample]Oo._cff import readFiles
from Alignment.OfflineValidation..oO[RelValSample]Oo._cff import secFiles
source = cms.Source ("PoolSource",
    fileNames = readFiles,
    secondaryFileNames = secFiles,
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
    fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

process.source = source

process.re_tracking_and_TP = cms.Sequence(process.mix*process.trackingParticles*
                                   process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.multiTrackValidator
                                   )

process.re_tracking = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*
                                   process.ckftracks*
                                   process.cutsRecoTracks*
                                   process.multiTrackValidator
                                   )

### final path and endPath
process.p = cms.Path(process.re_tracking)
"""

TrackSplittingTemplate="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("splitter")

# CMSSW.2.2.3

# message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_TrackSplitting_.oO[name]Oo.', 
        'cout')
)
## report only every 100th record
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')

# including global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# setting global tag
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."


###########################################
##necessary fix for the moment to avoid
##Assymmetric forward layers in TrackerException going through path p
##---- ScheduleExecutionFailure END
##an exception occurred during current event processing
##cms::Exception caught in EventProcessor and rethrown
##---- EventProcessorFailure END
############################################
#import CalibTracker.Configuration.Common.PoolDBESSource_cfi
from CondCore.DBCommon.CondDBSetup_cfi import *
#load the Global Position Rcd
process.globalPosition = cms.ESSource("PoolDBESSource", CondDBSetup,
                                  toGet = cms.VPSet(cms.PSet(
                                          record =cms.string('GlobalPositionRcd'),
                                          tag= cms.string('IdealGeometry')
                                          )),
                                  connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X')
                                  )
process.es_prefer_GPRcd = cms.ESPrefer("PoolDBESSource","globalPosition")
########################################## 


# track selectors and refitting
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

# including data...
process.load("Alignment.OfflineValidation..oO[superPointingDataset]Oo._cff")

## for craft SP skim v5
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_*_*_FU","drop *_*_*_HLT","drop *_MEtoEDMConverter_*_*","drop *_lumiProducer_*_REPACKER")
process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool( False )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
)


# magnetic field
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# adding geometries
from CondCore.DBCommon.CondDBSetup_cfi import *

# for craft
## tracker alignment for craft...............................................................
.oO[dbLoad]Oo.

.oO[APE]Oo.

## track hit filter.............................................................

# refit tracks first
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
      src = '.oO[TrackCollection]Oo.',
      TrajectoryInEvent = True,
      TTRHBuilder = "WithTrackAngle",
      NavigationSchool = ""
      )
      
process.FittingSmootherRKP5.EstimateCut = -1

# module configuration
# alignment track selector
process.AlignmentTrackSelector.src = "TrackRefitter1"
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.pMin   = 4.	
process.AlignmentTrackSelector.ptMax   = 9999.	
process.AlignmentTrackSelector.pMax   = 9999.	
process.AlignmentTrackSelector.etaMin  = -9999.
process.AlignmentTrackSelector.etaMax  = 9999.
process.AlignmentTrackSelector.nHitMin = 10
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.minHitsPerSubDet.inBPIX=4 ##skip tracks not passing the pixel
process.AlignmentTrackSelector.chi2nMax = 9999.
process.AlignmentTrackSelector.applyMultiplicityFilter = True
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
process.AlignmentTrackSelector.minHitsPerSubDet.inBPIX = 2
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]
process.KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut=30.0
process.KFFittingSmootherWithOutliersRejectionAndRK.MinNumberOfHits=4
#process.FittingSmootherRKP5.EstimateCut = 20.0
#process.FittingSmootherRKP5.MinNumberOfHits = 4

# configuration of the track spitting module
# new cuts allow for cutting on the impact parameter of the original track
process.load("RecoTracker.FinalTrackSelectors.cosmicTrackSplitter_cfi")
process.cosmicTrackSplitter.tracks = 'AlignmentTrackSelector'
process.cosmicTrackSplitter.tjTkAssociationMapTag = 'TrackRefitter1'
#process.cosmicTrackSplitter.excludePixelHits = False

#---------------------------------------------------------------------
# the output of the track hit filter are track candidates
# give them to the TrackProducer
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff")
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
     src = 'cosmicTrackSplitter',
     TrajectoryInEvent = True,
     TTRHBuilder = "WithTrackAngle"
)
# second refit
process.TrackRefitter2 = process.TrackRefitter1.clone(
         src = 'HitFilteredTracks'
         )

### Now adding the construction of global Muons
# what Chang did...
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.cosmicValidation = cms.EDAnalyzer("CosmicSplitterValidation",
	ifSplitMuons = cms.bool(False),
	ifTrackMCTruth = cms.bool(False),	
	checkIfGolden = cms.bool(False),	
    splitTracks = cms.InputTag("TrackRefitter2","","splitter"),
	splitGlobalMuons = cms.InputTag("muons","","splitter"),
	originalTracks = cms.InputTag("TrackRefitter1","","splitter"),
	originalGlobalMuons = cms.InputTag("muons","","Rec")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('.oO[outputFile]Oo.')
)

process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.AlignmentTrackSelector*process.cosmicTrackSplitter*process.HitFilteredTracks*process.TrackRefitter2*process.cosmicValidation)
"""

###
#                 Alternate Templates
###

def alternateTemplate( templateName, alternateTemplateName ):
   
    if not templateName in globals().keys():
        raise StandardError, "unkown template to replace %s"%templateName
    if not alternateTemplateName in globals().keys():
        raise StandardError, "unkown template to replace %s"%alternateTemplateName
    globals()[ templateName ] = globals()[ alternateTemplateName ]
    # = eval("configTemplates.%s"%"alternateTemplate")



###
otherTemplate = """
schum schum
"""

yResidualsOfflineValidation="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('LOGFILE_Offline_.oO[name]Oo.', 
        'cout')
)

 ## report only every 100th record
 ##process.MessageLogger.cerr.FwkReport.reportEvery = 100

    
 ##
 ## Alignment Track Selection
 ##
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'TrackRefitter1'
process.AlignmentTrackSelector.filter = True
process.AlignmentTrackSelector.applyBasicCuts = True
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

####  new FILTER
#-- new track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    369136710, 369136714, 402668822,
    # TOB
    436310989, 436310990, 436299301, 436299302,
    # TEC
    470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 14.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff.cosmictrackfinderP5.clone(
    src = 'TrackerTrackHitFilter'
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone(
  src = '.oO[TrackCollection]Oo.',
  TrajectoryInEvent = True,
  TTRHBuilder = "WithTrackAngle",
  NavigationSchool = ""
)

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'HitFilteredTracks')


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
#process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"

## LAYERWISE Lorentz Angle ###################

process.SiStripLorentzAngle = cms.ESSource("PoolDBESSource",
     BlobStreamerName = 
cms.untracked.string('TBufferBlobStreamingService'),
     DBParameters = cms.PSet(
         messageLevel = cms.untracked.int32(2),
         authenticationPath = 
cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
     ),
     timetype = cms.string('runnumber'),
     toGet = cms.VPSet(cms.PSet(
         record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLA_CRAFT_layers')
     )),
     connect = cms.string('sqlite_file:/afs/cern.ch/user/j/jdraeger/public/LA_object/LA_CRAFT_layers.db')
)
process.es_prefer_SiStripLorentzAngle = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngle")
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
process.load("Configuration/StandardSequences/MagneticField_38T_cff")

.oO[APE]Oo.

.oO[dbLoad]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..stripYResiduals = True
.oO[offlineValidationFileOutput]Oo.

 ##
 ## PATH
 ##
process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.TrackRefitter2*process.AlignmentTrackSelector*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)

"""

zeroAPETemplate="""
from CondCore.DBCommon.CondDBSetup_cfi import *
process.APE = cms.ESSource("PoolDBESSource",CondDBSetup,
                                        connect = cms.string('frontier://FrontierProd/CMS_COND_31X_FROM21X'),
                                        timetype = cms.string("runnumber"),
                                        toGet = cms.VPSet(
                                                          cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                                                                   tag = cms.string('TrackerIdealGeometryErrors210_mc')
                                                                   ))
                                        )
process.es_prefer_APE = cms.ESPrefer("PoolDBESSource", "APE")
"""

CosmicsOfflineValidation="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

#process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
reportEvery = cms.untracked.int32(1000) # every 1000th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
))
process.MessageLogger.statistics.append('cout')

#Physics declared
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'  

#-- Track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'



#-- Alignment Track Selection
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'HitFilteredTracks'
process.AlignmentTrackSelector.filter = True

.oO[TrackSelectionTemplate]Oo.

#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = 'TrackerTrackHitFilter',
###    TrajectoryInEvent = True,
     NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate"    
)

 ##
 ## Load and Configure TrackRefitter1
 ##


#############
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.TrackRefitter1 = process.TrackRefitterP5.clone(
    src = 'ALCARECOTkAlCosmicsCTF0T',
    TrajectoryInEvent = True,
     NavigationSchool = '',
    TTRHBuilder = "WithAngleAndTemplate"
)
process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'AlignmentTrackSelector',
   
    )


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."


.oO[LorentzAngleTemplate]Oo.
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
process.load("Configuration/StandardSequences/MagneticField_38T_cff")

.oO[dbLoad]Oo.

.oO[APE]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##

process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)
process.TFileService.fileName = '.oO[outputFile]Oo.'

 ##
 ## PATH
 ##
process.p = cms.Path(#process.triggerSelection*
process.offlineBeamSpot*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.AlignmentTrackSelector*process.TrackRefitter2*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)


"""


CosmicsAt0TOfflineValidation="""
import FWCore.ParameterSet.Config as cms

process = cms.Process("OfflineValidator") 
   
process.load("Alignment.OfflineValidation..oO[dataset]Oo._cff")

#process.source.inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*') # hack to get rid of the memory consumption problem in 2_2_X and beond
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

 ##
 ## Maximum number of Events
 ## 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(.oO[nEvents]Oo.)
 )

 ##   
 ## Messages & Convenience
 ##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
reportEvery = cms.untracked.int32(1000) # every 1000th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
))
process.MessageLogger.statistics.append('cout')


#-- Track hit filter
# TrackerTrackHitFilter takes as input the tracks/trajectories coming out from TrackRefitter1
process.load("RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff")
process.TrackerTrackHitFilter.src = 'TrackRefitter1'

#-- Alignment Track Selection
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'HitFilteredTracks'
process.AlignmentTrackSelector.filter = True

.oO[TrackSelectionTemplate]Oo.
# Override the pmin setting since not meaningful with B=0T
process.AlignmentTrackSelector.pMin    = 4.

#### momentum constraint for 0T
# First momentum constraint
process.load("RecoTracker.TrackProducer.MomentumConstraintProducer_cff")
import RecoTracker.TrackProducer.MomentumConstraintProducer_cff
process.AliMomConstraint1 = RecoTracker.TrackProducer.MomentumConstraintProducer_cff.MyMomConstraint.clone()
process.AliMomConstraint1.src = '.oO[TrackCollection]Oo.'
process.AliMomConstraint1.fixedMomentum = 5.0
process.AliMomConstraint1.fixedMomentumError = 0.005

# Second momentum constraint
#process.load("RecoTracker.TrackProducer.MomentumConstraintProducer_cff")
#import RecoTracker.TrackProducer.MomentumConstraintProducer_cff
#process.AliMomConstraint2 = RecoTracker.TrackProducer.MomentumConstraintProducer_cff.MyMomConstraint.clone()
#process.AliMomConstraint2.src = 'AlignmentTrackSelector'
#process.AliMomConstraint2.fixedMomentum = 5.0
#process.AliMomConstraint2.fixedMomentumError = 0.005
   


#now we give the TrackCandidate coming out of the TrackerTrackHitFilter to the track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
process.HitFilteredTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = 'TrackerTrackHitFilter'
###    ,
###    TrajectoryInEvent = True,
###    TTRHBuilder = "WithAngleAndTemplate"    
)

 ##
 ## Load and Configure TrackRefitter1
 ##

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

#############
# parameters for TrackRefitter
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter1 = process.TrackRefitterP5.clone(
   src =  '.oO[TrackCollection]Oo.'#'AliMomConstraint1',
   TrajectoryInEvent = True,
   TTRHBuilder = "WithAngleAndTemplate",
   NavigationSchool = "",
   constraint = 'momentum', ### SPECIFIC FOR CRUZET
   srcConstr='AliMomConstraint1' ### SPECIFIC FOR CRUZET$works only with tag V02-10-02 TrackingTools/PatternTools / or CMSSW >=31X
)

process.TrackRefitter2 = process.TrackRefitter1.clone(
    src = 'AlignmentTrackSelector',
    srcConstr='AliMomConstraint1',
    constraint = 'momentum' ### SPECIFIC FOR CRUZET
)


 ##
 ## Get the BeamSpot
 ##
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
 
 ##
 ## GlobalTag Conditions (if needed)
 ##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ".oO[GlobalTag]Oo."
process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"

.oO[LorentzAngleTemplate]Oo.
  
 ##
 ## Geometry
 ##
process.load("Configuration.StandardSequences.Geometry_cff")
 
 ##
 ## Magnetic Field
 ##
#process.load("Configuration/StandardSequences/MagneticField_38T_cff")
process.load("Configuration.StandardSequences.MagneticField_0T_cff") # 0T runs

.oO[dbLoad]Oo.

.oO[APE]Oo.

## to apply misalignments
#TrackerDigiGeometryESModule.applyAlignment = True
   
 ##
 ## Load and Configure OfflineValidation
 ##
process.load("Alignment.OfflineValidation.TrackerOfflineValidation_.oO[offlineValidationMode]Oo._cff")
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..Tracks = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..trajectoryInput = 'TrackRefitter2'
process.TrackerOfflineValidation.oO[offlineValidationMode]Oo..moduleLevelHistsTransient = cms.bool(.oO[offlineModuleLevelHistsTransient]Oo.)
process.TFileService.fileName = '.oO[outputFile]Oo.'

 ##
 ## PATH
 ##

process.p = cms.Path(process.offlineBeamSpot*process.AliMomConstraint1*process.TrackRefitter1*process.TrackerTrackHitFilter*process.HitFilteredTracks
                     *process.AlignmentTrackSelector*process.TrackRefitter2*process.seqTrackerOfflineValidation.oO[offlineValidationMode]Oo.)


"""
TrackSelectionCosmicsDuringCollisions = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

#######################################
##Trigger settings for Cosmics during collisions
#######################################
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.L1T1=process.hltLevel1GTSeed.clone()
process.L1T1.L1TechTriggerSeeding = cms.bool(True)
process.L1T1.L1SeedsLogicalExpression=cms.string('25') 
process.hltHighLevel = cms.EDFilter("HLTHighLevel",
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    HLTPaths = cms.vstring('HLT_TrackerCosmics'),
    eventSetupPathsKey = cms.string(''),
    andOr = cms.bool(False),
    throw = cms.bool(True)
)


process.triggerSelection=cms.Sequence(process.L1T1*process.hltHighLevel)
"""
TrackSelectionCosmicsData = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True
#process.TrackerTrackHitFilter.PxlCorrClusterChargeCut = 10000.0
process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""
TrackSelectionCosmicsDataBPIX = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
process.AlignmentTrackSelector.minHitsPerSubDet = cms.PSet(
         inTEC = cms.int32(0),
         inTOB = cms.int32(0),
         inFPIX = cms.int32(0),
        inTID = cms.int32(0),
         inBPIX = cms.int32(1),
         inTIB = cms.int32(0),
         inPIXEL = cms.int32(0),
         inTIDplus = cms.int32(0),
         inTIDminus = cms.int32(0),
         inTECplus = cms.int32(0),
         inTECminus = cms.int32(0),
         inFPIXplus = cms.int32(0),
         inFPIXminus = cms.int32(0),
         inENDCAP = cms.int32(0),
         inENDCAPplus = cms.int32(0),
         inENDCAPminus = cms.int32(0),
     )
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""
TrackSelectionCosmicsDataFPIXplus = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
process.AlignmentTrackSelector.minHitsPerSubDet = cms.PSet(
         inTEC = cms.int32(0),
         inTOB = cms.int32(0),
         inFPIX = cms.int32(0),
        inTID = cms.int32(0),
         inBPIX = cms.int32(0),
         inTIB = cms.int32(0),
         inPIXEL = cms.int32(0),
         inTIDplus = cms.int32(0),
         inTIDminus = cms.int32(0),
         inTECplus = cms.int32(0),
         inTECminus = cms.int32(0),
         inFPIXplus = cms.int32(1),
         inFPIXminus = cms.int32(0),
         inENDCAP = cms.int32(0),
         inENDCAPplus = cms.int32(0),
         inENDCAPminus = cms.int32(0),
     )
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""
TrackSelectionCosmicsDataFPIXminus = """
##### For Tracks:
process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 4.
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
process.AlignmentTrackSelector.minHitsPerSubDet = cms.PSet(
         inTEC = cms.int32(0),
         inTOB = cms.int32(0),
         inFPIX = cms.int32(0),
        inTID = cms.int32(0),
         inBPIX = cms.int32(0),
         inTIB = cms.int32(0),
         inPIXEL = cms.int32(0),
         inTIDplus = cms.int32(0),
         inTIDminus = cms.int32(0),
         inTECplus = cms.int32(0),
         inTECminus = cms.int32(0),
         inFPIXplus = cms.int32(0),
         inFPIXminus = cms.int32(1),
         inENDCAP = cms.int32(0),
         inENDCAPplus = cms.int32(0),
         inENDCAPminus = cms.int32(0),
     )
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 18.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.35 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

process.triggerSelection=cms.Sequence(process.hltPhysicsDeclared)
"""
TrackSelectionMinBiasData = """
##### For Tracks:collisions taken in deco mode

process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 3
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 0.65
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,

    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.17 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

##############
##Trigger sequence
#############
#bit 0 is selecting bunch crossing
#bit 40 MinBias trigger


process.triggerSelection=cms.Sequence(process.oneGoodVertexFilter)

"""
TrackSelectionIsolatedMuons = """
##### For Tracks:collisions taken in deco mode

process.AlignmentTrackSelector.applyBasicCuts = True
# Note that pMin is overridden and set to zero in
# the offlineTemplate0T
process.AlignmentTrackSelector.pMin    = 0
process.AlignmentTrackSelector.pMax    = 9999.
process.AlignmentTrackSelector.ptMin   = 3.
process.AlignmentTrackSelector.ptMax   = 9999.
process.AlignmentTrackSelector.etaMin  = -999.
process.AlignmentTrackSelector.etaMax  = 999.
process.AlignmentTrackSelector.nHitMin = 8
process.AlignmentTrackSelector.nHitMin2D = 2
process.AlignmentTrackSelector.chi2nMax = 999.
process.AlignmentTrackSelector.applyMultiplicityFilter = False
process.AlignmentTrackSelector.maxMultiplicity = 1
process.AlignmentTrackSelector.applyNHighestPt = False
process.AlignmentTrackSelector.nHighestPt = 1
process.AlignmentTrackSelector.seedOnlyFrom = 0 
process.AlignmentTrackSelector.applyIsolationCut = False
process.AlignmentTrackSelector.minHitIsolation = 0.8
process.AlignmentTrackSelector.applyChargeCheck = False
process.AlignmentTrackSelector.minHitChargeStrip = 50.
#process.AlignmentTrackSelector.trackQualities = ["highPurity"]
#process.AlignmentTrackSelector.iterativeTrackingSteps = ["iter1","iter2"]

##### For Hits:
process.TrackerTrackHitFilter.useTrajectories= True  # this is needed only if you require some selections; but it will work even if you don't ask for them
process.TrackerTrackHitFilter.minimumHits = 8
process.TrackerTrackHitFilter.commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC")
process.TrackerTrackHitFilter.detsToIgnore = [
     # see https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/484.html
    # TIB / TID
    #369136710, 369136714, 402668822,
    # TOB
    #436310989, 436310990, 436299301, 436299302,
    # TEC
    #470340521, 470063045, 470063046, 470114669, 470114670, 470161093, 470161094, 470164333, 470164334, 470312005, 470312006, 470312009, 470067405, 470067406, 470128813
]
process.TrackerTrackHitFilter.replaceWithInactiveHits = True
process.TrackerTrackHitFilter.stripAllInvalidHits = False
process.TrackerTrackHitFilter.rejectBadStoNHits = True
process.TrackerTrackHitFilter.StoNcommands = cms.vstring("ALL 12.0")
process.TrackerTrackHitFilter.rejectLowAngleHits= True
process.TrackerTrackHitFilter.TrackAngleCut= 0.17 # in rads, starting from the module surface
process.TrackerTrackHitFilter.usePixelQualityFlag= True

##############
##Trigger sequence
#############
#bit 0 is selecting bunch crossing
#bit xy for muons trigger


process.triggerSelection=cms.Sequence(process.bptxAnd)

"""
