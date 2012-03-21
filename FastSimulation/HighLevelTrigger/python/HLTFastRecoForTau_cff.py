import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.HLTPixelTracksProducer_cfi

hltRegionalPixelTracks = FastSimulation.Tracking.HLTPixelTracksProducer_cfi.hltPixelTracks.clone()
hltRegionalPixelTracks.FilterPSet.ptMin = 0.1
hltRegionalPixelTracks.RegionFactoryPSet.ComponentName = "L3MumuTrackingRegion"
hltRegionalPixelTracks.RegionFactoryPSet.RegionPSet = cms.PSet(
    originRadius = cms.double( 1.0 ),
    ptMin = cms.double( 0.5 ),
    originHalfLength = cms.double( 15.0 ),
    vertexZDefault = cms.double( 0.0 ),
    vertexSrc = cms.string( "hltDisplacedmumuVtxProducerDoubleMuTau2Mu" ),
    deltaEtaRegion = cms.double( 0.5 ),
    deltaPhiRegion = cms.double( 0.5 ),
    TrkSrc = cms.InputTag( "hltL3Muons" ),
    UseVtxTks = cms.bool( False )
)



# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltTau3MuCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltTau3MuCkfTrackCandidates.SeedProducer = cms.InputTag("hltTau3MuPixelSeedsFromPixelTracks")
hltTau3MuCkfTrackCandidates.TrackProducers = []
hltTau3MuCkfTrackCandidates.SeedCleaning = True
hltTau3MuCkfTrackCandidates.SplitHits = False

# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltTau3MuCtfWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltTau3MuCtfWithMaterialTracks.src = 'hltTau3MuCkfTrackCandidates'
hltTau3MuCtfWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltTau3MuCtfWithMaterialTracks.Fitter = 'KFFittingSmoother'
hltTau3MuCtfWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

