######################################################################
######################################################################
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
# process.GlobalTag.connect="frontier://FrontierProd/CMS_COND_31X_GLOBALTAG"
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
.oO[datasetDefinition]Oo.

## for craft SP skim v5
process.source.inputCommands = cms.untracked.vstring("keep *","drop *_*_*_FU","drop *_*_*_HLT","drop *_MEtoEDMConverter_*_*","drop *_lumiProducer_*_REPACKER")
process.source.dropDescendantsOfDroppedBranches = cms.untracked.bool( False )


# magnetic field
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# adding geometries
from CondCore.DBCommon.CondDBSetup_cfi import *

# for craft
## tracker alignment for craft...............................................................
.oO[condLoad]Oo.

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

