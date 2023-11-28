import FWCore.ParameterSet.Config as cms

process = cms.Process("testAlignmentStats") 

###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8

###################################################################
# Messages
###################################################################

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.enable = False
process.MessageLogger.GeoInfo=dict()
process.MessageLogger.AlignmentStats=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(10)
                                   ),
    GeoInfo          = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    AlignmentStats    = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

####################################################################
# Get the Magnetic Field
####################################################################
process.load('Configuration.StandardSequences.MagneticField_cff')

###################################################################
# Standard loads
###################################################################
process.load("Configuration.Geometry.GeometryRecoDB_cff")

####################################################################
# Produce the Transient Track Record in the event
####################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

###################################################################
# Event source and run selection
###################################################################
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_TTBarPU
process.source = cms.Source("PoolSource",
                            fileNames = filesDefaultMC_TTBarPU,
                            duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
                            )

runboundary = 1
process.source.firstRun = cms.untracked.uint32(int(runboundary))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

####################################################################
# Get the BeamSpot
####################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.myBeamSpot = process.offlineBeamSpot.clone()

####################################################################
#1: first refit to the tracks, needed for getting the Traj
####################################################################
from TrackingTools.TrackFitters.RungeKuttaFitters_cff import *
process.FittingSmootherCustomised = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(ComponentName = 'FittingSmootherCustomised',
                                                                                                                                       EstimateCut=18.0,
                                                                                                                                       MinNumberOfHits=6)
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitterCTF1 = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    constraint = "",
    src='generalTracks',
    NavigationSchool = '',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True,
    beamSpot='myBeamSpot')

####################################################################
# 2b: apply NEW hit filter. Does not work with CosmicTF tracks !
####################################################################
from RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff import *
process.AlignmentHitFilterCTF = RecoTracker.FinalTrackSelectors.TrackerTrackHitFilter_cff.TrackerTrackHitFilter.clone(
    src = 'TrackRefitterCTF1',
    commands = cms.vstring("keep PXB","keep PXE","keep TIB","keep TID","keep TOB","keep TEC"),
    minimumHits = 6,
    replaceWithInactiveHits = True,
    stripAllInvalidHits = False,
    rejectBadStoNHits = True,
    StoNcommands = cms.vstring("ALL 18.0"),
    useTrajectories= True,
    rejectLowAngleHits= True,
    TrackAngleCut= 0.17,
    usePixelQualityFlag= True,
    PxlCorrClusterChargeCut=10000.0)

####################################################################
# 3: produce track after NEW track hit filter
####################################################################
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
process.ctfProducerCustomisedCTF = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff.ctfWithMaterialTracks.clone(
    src = 'AlignmentHitFilterCTF',
    beamSpot='myBeamSpot',
    # Fitter = 'FittingSmootherCustomised',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True)

####################################################################
# 4: apply track selections on the refitted tracks
####################################################################
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
process.ALCARECOTkAlMinBiasSkimmed = AlignmentTrackSelector.clone(
    src= 'ctfProducerCustomisedCTF',
    ptMin=1.5, # already in ALCARECO cfg
    ptMax=9999.0,
    pMin=3.0,
    pMax=9999.0,
    etaMin=-2.4,  # already in ALCARECO cfg
    etaMax=2.4,   # already in ALCARECO cfg
    nHitMin=8,
    nHitMin2D=2,
    chi2nMax=6.0
    ### others which aren't used
    # minHitsPerSubDet.inTIB = 0
    # minHitsPerSubDet.inBPIX = 1
    )

process.TrackRefitterCTF2 = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
    constraint = "",
    src='ALCARECOTkAlMinBiasSkimmed',
    TTRHBuilder = 'WithAngleAndTemplate',
    TrajectoryInEvent = True,
    NavigationSchool = '',
    beamSpot='myBeamSpot',
    # EstimateCut=15.0,
    # MinNumberOfHits=6
    # Fitter='FittingSmootherCustomised'
    ) 

####################################################################
# 5: Overlap tagger
####################################################################
from Alignment.TrackerAlignment.TkAlCaOverlapTagger_cff import *
process.OverlapAssoMapCTF = OverlapTagger.clone(
    # src='ALCARECOTkAlCosmicsCTFSkimmed'
    src='TrackRefitterCTF2',
    #Clustersrc='ALCARECOTkAlCosmicsCTF0T'
    Clustersrc='ALCARECOTkAlMinBiasSkimmed'#the track selector produces a new collection of Clusters!
)


####################################################################
# 6: counts
####################################################################
from Alignment.CommonAlignmentMonitor.AlignmentStats_cff import *
process.NewStatsCTF = AlignmentStats.clone(
    #  src='OverlapAssoMap',
    src='TrackRefitterCTF2',
    OverlapAssoMap='OverlapAssoMapCTF',
    keepTrackStats = False,
    keepHitStats = True,
    TrkStatsFileName='TracksStatisticsCTF.root',
    HitStatsFileName='HitMapsCTF.root',
    TrkStatsPrescale= 1                            
    )

##________________________________Sequences____________________________________
process.seqALCARECOTkAlMinBiasSkimmed = cms.Sequence(process.myBeamSpot *
                                                     process.offlineBeamSpot *
                                                     process.MeasurementTrackerEvent *
                                                     process.TrackRefitterCTF1 *
                                                     process.AlignmentHitFilterCTF *
                                                     process.ctfProducerCustomisedCTF *
                                                     process.ALCARECOTkAlMinBiasSkimmed *
                                                     process.TrackRefitterCTF2 *
                                                     process.OverlapAssoMapCTF *
                                                     process.NewStatsCTF)

process.p2 = cms.Path(process.seqALCARECOTkAlMinBiasSkimmed)
