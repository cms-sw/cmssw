import FWCore.ParameterSet.Config as cms

hltPhase2SiPixelRecHitsSoA = cms.EDProducer('SiPixelRecHitAlpakaPhase2@alpaka',
    beamSpot = cms.InputTag('hltPhase2OnlineBeamSpotDevice'),
    src = cms.InputTag('hltPhase2SiPixelClustersSoA'),
    CPE = cms.string('PixelCPEFastParamsPhase2'),
    mightGet = cms.optional.untracked.vstring,
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
