import FWCore.ParameterSet.Config as cms

hltPhase2SiPixelRecHitsSoA = cms.EDProducer('SiPixelRecHitAlpakaPhase2@alpaka',
    beamSpot = cms.InputTag('hltPhase2OnlineBeamSpotDevice'),
    src = cms.InputTag('hltPhase2SiPixelClustersSoA'),
    CPE = cms.string('PixelCPEFastParamsPhase2'),
    mightGet = cms.optional.untracked.vstring,
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

# Same pixel hits, with errors from per-module genError parameters instead of the cluster-size table.
from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
_hltPhase2SiPixelRecHitsSoAWithStubs = cms.EDProducer('SiPixelRecHitAlpakaPhase2OTStubs@alpaka',
    beamSpot = cms.InputTag('hltPhase2OnlineBeamSpotDevice'),
    src = cms.InputTag('hltPhase2SiPixelClustersSoA'),
    CPE = cms.string('PixelCPEFastParamsPhase2OTStubs'),
    mightGet = cms.optional.untracked.vstring,
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
phase2CAStubs.toReplaceWith(hltPhase2SiPixelRecHitsSoA, _hltPhase2SiPixelRecHitsSoAWithStubs)
