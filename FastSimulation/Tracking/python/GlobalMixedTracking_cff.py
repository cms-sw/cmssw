import FWCore.ParameterSet.Config as cms

# Global Mixed seeding
from FastSimulation.Tracking.GlobalMixedSeedProducer_cff import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
# TrackCandidates
globalMixedGSTrackCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# reco::Tracks
globalMixedGSWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
# Merging
ctfGSWithMaterialTracks = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("globalMixedGSTrackCandidates"), cms.InputTag("globalMixedGSWithMaterialTracks"))
)

# The sequence
ctfGSTracking = cms.Sequence(globalMixedGSSeeds*globalMixedGSTrackCandidates*globalMixedGSWithMaterialTracks*ctfGSWithMaterialTracks)
globalMixedGSTrackCandidates.SeedProducer = cms.InputTag("globalMixedGSSeeds","GlobalMixed")
globalMixedGSTrackCandidates.TrackProducer = 'globalPixelGSWithMaterialTracks'
globalMixedGSWithMaterialTracks.src = 'globalMixedGSTrackCandidates'
globalMixedGSWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
globalMixedGSWithMaterialTracks.Fitter = 'KFFittingSmoother'
globalMixedGSWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

