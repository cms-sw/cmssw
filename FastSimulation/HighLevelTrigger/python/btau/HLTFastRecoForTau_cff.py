import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
ctfWithMaterialTracksL25SingleTau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(5.0)
)

ctfWithMaterialTracksL25SingleTauMET = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(5.0)
)

hltCtfWithMaterialTracksL25ElectronTau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(5.0)
)

ctfWithMaterialTracksL3SingleTau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(1.0)
)

ctfWithMaterialTracksL3SingleTauMET = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(1.0)
)

hltCtfWithMaterialTracksL3ElectronTau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelGSWithMaterialTracks")),
    ptMin = cms.untracked.double(1.0)
)

ckfTracksL25SingleTau = cms.Sequence(globalPixelGSTracking+ctfWithMaterialTracksL25SingleTau)
ckfTracksL25SingleTauMET = cms.Sequence(globalPixelGSTracking+ctfWithMaterialTracksL25SingleTauMET)
hltTracksL25ElectronTau = cms.Sequence(globalPixelGSTracking*hltCtfWithMaterialTracksL25ElectronTau)
ckfTracksL3SingleTau = cms.Sequence(globalPixelGSTracking*ctfWithMaterialTracksL3SingleTau)
ckfTracksL3SingleTauMET = cms.Sequence(globalPixelGSTracking+ctfWithMaterialTracksL3SingleTauMET)
hltTracksL3ElectronTau = cms.Sequence(globalPixelGSTracking+hltCtfWithMaterialTracksL3ElectronTau)
doubleTauL1SeedFilter.L1MuonCollectionTag = 'l1ParamMuons'
singleTauL1SeedFilter.L1MuonCollectionTag = 'l1ParamMuons'
singleTauMETL1SeedFilter.L1MuonCollectionTag = 'l1ParamMuons'
doubleTauL1SeedFilter.L1GtObjectMapTag = 'gtDigis'
singleTauL1SeedFilter.L1GtObjectMapTag = 'gtDigis'
singleTauMETL1SeedFilter.L1GtObjectMapTag = 'gtDigis'

