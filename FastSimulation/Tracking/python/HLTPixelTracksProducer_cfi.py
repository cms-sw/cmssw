import FWCore.ParameterSet.Config as cms

import FastSimulation.HighLevelTrigger.DummyModule_cfi

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *

hltPixelTracks = cms.EDProducer("PixelTracksProducer",
    FitterPSet = cms.PSet(
        ComponentName = cms.string('PixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('WithoutRefit')
    ),
    SeedProducer = cms.InputTag("pixelTripletSeeds"),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    FilterPSet = cms.PSet(
        nSigmaInvPtTolerance = cms.double(0.0),
        nSigmaTipMaxTolerance = cms.double(0.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        chi2 = cms.double(1000.0),
        ptMin = cms.double(0.1),
        tipMax = cms.double(1.0)
    )
)


hltPixelTracksReg = hltPixelTracks.clone()
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
    whereToUseMeasurementTracker = cms.string("Never"))

hltPixelTracksHybrid = hltPixelTracks.clone()
hltPixelTracksL3Muon = hltPixelTracks.clone()
hltPixelTracksGlbTrkMuon = hltPixelTracks.clone()
hltPixelTracksHighPtTkMuIso = hltPixelTracks.clone()
hltPixelTracksForPhotons = hltPixelTracks.clone()
hltPixelTracksForEgamma = hltPixelTracks.clone()
hltPixelTracksElectrons = hltPixelTracks.clone()
hltPixelTracksForHighPt = hltPixelTracks.clone()
hltHighPtPixelTracks = hltPixelTracks.clone()

hltPixelTracksForNoPU = hltPixelTracks.clone()
hltDummyLocalPixel = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
HLTDoLocalPixelSequenceRegForNoPU = cms.Sequence(hltDummyLocalPixel)

hltFastPixelHitsVertexVHbb = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltFastPixelTracksVHbb = hltPixelTracks.clone()
hltFastPixelTracksRecoverVHbb = hltPixelTracks.clone()

hltFastPrimaryVertex = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltFastPVPixelVertexFilter = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltFastPVPixelTracks = hltPixelTracks.clone()
hltFastPVPixelTracksRecover = hltPixelTracks.clone()

hltPixelLayerPairs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltPixelLayerTriplets = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltPixelLayerTripletsReg = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltPixelLayerTripletsHITHB = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltPixelLayerTripletsHITHE = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltMixedLayerPairs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
