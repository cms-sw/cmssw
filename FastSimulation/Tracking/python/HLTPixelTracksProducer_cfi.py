import FWCore.ParameterSet.Config as cms

import FastSimulation.HighLevelTrigger.DummyModule_cfi

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *

hltPixelTracks = cms.EDProducer("PixelTracksProducer",
    FitterPSet = cms.PSet(
        ComponentName = cms.string('PixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('WithoutRefit')
    ),
    SeedProducer = cms.InputTag("pixelTripletSeeds","PixelTriplet"),
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
