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
    vertexSrc = cms.InputTag( "hltDisplacedmumuVtxProducerDoubleMuTau2Mu" ),
    deltaEtaRegion = cms.double( 0.5 ),
    deltaPhiRegion = cms.double( 0.5 ),
    TrkSrc = cms.InputTag( "hltL3Muons" ),
    UseVtxTks = cms.bool( False ),
    howToUseMeasurementTracker = cms.string("Never"),
)

hltPixelTracksReg = FastSimulation.Tracking.HLTPixelTracksProducer_cfi.hltPixelTracks.clone()
hltPixelTracksReg.FilterPSet.ptMin = 0.1
hltPixelTracksReg.FilterPSet.chi2 = 50.
hltPixelTracksReg.RegionFactoryPSet.ComponentName = "CandidateSeededTrackingRegionsProducer"
hltPixelTracksReg.RegionFactoryPSet.RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 0.9 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        mode = cms.string( "BeamSpotSigma" ),
        input = cms.InputTag( "hltL2TausForPixelIsolation" ),
        maxNRegions = cms.int32( 10 ),
        vertexCollection = cms.InputTag( "" ),
        maxNVertices = cms.int32( 1 ),
        zErrorBeamSpot = cms.double( 24.2 ),
        deltaEta = cms.double( 0.5 ),
        deltaPhi = cms.double( 0.5 ),
        nSigmaZVertex = cms.double( 3.0 ),
        zErrorVertex = cms.double( 0.2 ),
        nSigmaZBeamSpot = cms.double( 4.0 ),
        whereToUseMeasurementTracker = cms.string("Never"),
)



# CKFTrackCandidateMaker
import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltTau3MuCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
hltTau3MuCkfTrackCandidates.src = cms.InputTag("hltTau3MuPixelSeedsFromPixelTracks")
hltTau3MuCkfTrackCandidates.SplitHits = False



# CTF track fit with material
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi

hltTau3MuCtfWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltTau3MuCtfWithMaterialTracks.src = 'hltTau3MuCkfTrackCandidates'
hltTau3MuCtfWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
hltTau3MuCtfWithMaterialTracks.Fitter = 'KFFittingSmoother'
hltTau3MuCtfWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

