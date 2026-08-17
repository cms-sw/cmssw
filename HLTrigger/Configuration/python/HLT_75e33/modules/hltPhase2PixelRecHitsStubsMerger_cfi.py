import FWCore.ParameterSet.Config as cms

hltPhase2PixelRecHitsStubsMerger = cms.EDProducer('SiPixelRecHitsStubsMerger@alpaka',
  pixelRecHitsSoA = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
  stubsSoA = cms.InputTag('hltOTStubProducer'),
  otRecHitsSoA = cms.InputTag('hltPixelSeedingOTRecHitsSoA'),
  mightGet = cms.optional.untracked.vstring,
  alpaka = cms.untracked.PSet(
    backend = cms.untracked.string('')
  )
)
