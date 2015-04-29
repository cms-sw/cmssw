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



hltFastPixelHitsVertex = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltFastPixelTracks = hltPixelTracks.clone()
hltFastPixelTracksRecover = hltPixelTracks.clone()

hltFastPrimaryVertexbbPhi = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltPixelTracksFastPVbbPhi = hltPixelTracks.clone()
hltPixelTracksRecoverbbPhi = hltPixelTracks.clone()

hltPixelTracksReg = hltPixelTracks.clone()
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
