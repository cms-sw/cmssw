import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
# L2 seeds from L1 input
# module hltL2MuonSeeds = L2MuonSeeds from "RecoMuon/L2MuonSeedGenerator/data/L2MuonSeeds.cfi"
# replace hltL2MuonSeeds.GMTReadoutCollection = l1extraParticles
# replace hltL2MuonSeeds.InputObjects = l1extraParticles
# L3 regional reconstruction
from FastSimulation.Muons.L3Muons_cff import *
#import FastSimulation.Muons.L3Muons_cfi
#hltL3Muonstemp = FastSimulation.Muons.L3Muons_cfi.L3Muons.clone()
#hltL3Muonstemp.L3TrajBuilderParameters.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
#hltL3Muonstemp.L3TrajBuilderParameters.TrackerRecHitBuilder = 'WithoutRefit'
#hltL3Muonstemp.TrackLoaderParameters.beamSpot = cms.InputTag("offlineBeamSpot")

# L3 regional seeding, candidating, tracking
#--the two below have to be picked up from confDB: 
# from FastSimulation.Muons.TSGFromL2_cfi import *
# from FastSimulation.Muons.HLTL3TkMuons_cfi import *
#from FastSimulation.Muons.TrackCandidateFromL2_cfi import *
#from FastSimulation.Muons.TSGFromL2_cfi import *
#hltL3TrajectorySeed = FastSimulation.Muons.TSGFromL2_cfi.hltL3TrajectorySeed.clone()

import FastSimulation.Muons.TSGFromL2_cfi as TSG
#from FastSimulation.Muons.TSGFromL2_cfi import OIStatePropagators as OIProp
from FastSimulation.Muons.TSGFromL2_cfi import OIHitPropagators as OIHProp
## Make three individual seeders
## OIState can be taken directly from configuration
#hltL3TrajSeedOIState = TSG.l3seeds("OIState")
#hltL3TrajSeedOIState.ServiceParameters.Propagators = cms.untracked.vstring()
#OIProp(hltL3TrajSeedOIState,hltL3TrajSeedOIState.TkSeedGenerator)
hltL3TrajSeedOIHit = TSG.l3seeds("OIHitCascade")
hltL3TrajSeedOIHit.ServiceParameters.Propagators = cms.untracked.vstring()
OIHProp(hltL3TrajSeedOIHit,hltL3TrajSeedOIHit.TkSeedGenerator.iterativeTSG)
hltL3TrajSeedIOHit = TSG.l3seeds("IOHitCascade")

## Make one TrackCand for each seeder
from FastSimulation.Muons.TrackCandidateFromL2_cfi import *
hltL3TrackCandidateFromL2OIState = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2OIState.SeedProducer = "hltL3TrajSeedOIState"
hltL3TrackCandidateFromL2OIHit = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2OIHit.SeedProducer = "hltL3TrajSeedOIHit"    
hltL3TrackCandidateFromL2IOHit = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2IOHit.SeedProducer = "hltL3TrajSeedIOHit"
hltL3TrackCandidateFromL2NoVtx = FastSimulation.Muons.TrackCandidateFromL2_cfi.hltL3TrackCandidateFromL2.clone()
hltL3TrackCandidateFromL2NoVtx.SeedProducer = "hltL3TrajectorySeedNoVtx"


# (Not-so) Regional Tracking - needed because the TrackCandidateProducer needs the seeds 
from FastSimulation.Tracking.GlobalPixelTracking_cff import *


# Seeds (just clone the hltMuTrackSeeds with a different InputVertexCollection, for now):
hltJpsiTkPixelSeedFromL3Candidate = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    InputVertexCollection = cms.InputTag( "hltDisplacedmumuVtxProducerDoubleMu4JpsiTk" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltL3Muons" ),
    originRadius = cms.double( 1.0 )
)


# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltMuCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltMuCkfTrackCandidates.SeedProducer = cms.InputTag("hltMuTrackSeeds")
hltMuCkfTrackCandidates.TrackProducers = []
hltMuCkfTrackCandidates.SeedCleaning = True
hltMuCkfTrackCandidates.SplitHits = False

hltMuTrackJpsiCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltMuTrackJpsiCkfTrackCandidates.SeedProducer = cms.InputTag("hltMuTrackJpsiTrackSeeds")
hltMuTrackJpsiCkfTrackCandidates.TrackProducers = []
hltMuTrackJpsiCkfTrackCandidates.SeedCleaning = True
hltMuTrackJpsiCkfTrackCandidates.SplitHits = False

hltMuTrackJpsiEffCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltMuTrackJpsiEffCkfTrackCandidates.SeedProducer = cms.InputTag("hltMuTrackJpsiTrackSeeds")
hltMuTrackJpsiEffCkfTrackCandidates.TrackProducers = []
hltMuTrackJpsiEffCkfTrackCandidates.SeedCleaning = True
hltMuTrackJpsiEffCkfTrackCandidates.SplitHits = False

hltCkfTrackCandidatesJpsiTk = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltCkfTrackCandidatesJpsiTk.SeedProducer = cms.InputTag("hltJpsiTkPixelSeedFromL3Candidate")
hltCkfTrackCandidatesJpsiTk.TrackProducers = []
hltCkfTrackCandidatesJpsiTk.SeedCleaning = True
hltCkfTrackCandidatesJpsiTk.SplitHits = False


# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltMuCtfTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltMuCtfTracks.src = 'hltMuCkfTrackCandidates'
hltMuCtfTracks.TTRHBuilder = 'WithoutRefit'
hltMuCtfTracks.Fitter = 'KFFittingSmoother'
hltMuCtfTracks.Propagator = 'PropagatorWithMaterial'

hltMuTrackJpsiCtfTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltMuTrackJpsiCtfTracks.src = 'hltMuTrackJpsiCkfTrackCandidates'
hltMuTrackJpsiCtfTracks.TTRHBuilder = 'WithoutRefit'
hltMuTrackJpsiCtfTracks.Fitter = 'KFFittingSmoother'
hltMuTrackJpsiCtfTracks.Propagator = 'PropagatorWithMaterial'

hltMuTrackJpsiEffCtfTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltMuTrackJpsiEffCtfTracks.src = 'hltMuTrackJpsiEffCkfTrackCandidates'
hltMuTrackJpsiEffCtfTracks.TTRHBuilder = 'WithoutRefit'
hltMuTrackJpsiEffCtfTracks.Fitter = 'KFFittingSmoother'
hltMuTrackJpsiEffCtfTracks.Propagator = 'PropagatorWithMaterial'

hltCtfWithMaterialTracksJpsiTk = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltCtfWithMaterialTracksJpsiTk.src = 'hltCkfTrackCandidatesJpsiTk'
hltCtfWithMaterialTracksJpsiTk.TTRHBuilder = 'WithoutRefit'
hltCtfWithMaterialTracksJpsiTk.Fitter = 'KFFittingSmoother'
hltCtfWithMaterialTracksJpsiTk.Propagator = 'PropagatorWithMaterial'


#hltMuTrackSeedsSequence = cms.Sequence(globalPixelTracking+
#                                     cms.SequencePlaceholder("hltMuTrackSeeds"))
#
#HLTMuTrackingSequence = cms.Sequence(hltMuCkfTrackCandidates+
#                                     hltMuCtfTracks+
#                                     cms.SequencePlaceholder("hltMuTracking"))


# L3 muon isolation sequence
from FastSimulation.Tracking.HLTPixelTracksProducer_cfi import *
HLTRegionalCKFTracksForL3Isolation = cms.Sequence( hltPixelTracks)

