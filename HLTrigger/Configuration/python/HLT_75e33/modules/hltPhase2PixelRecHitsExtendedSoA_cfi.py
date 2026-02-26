import FWCore.ParameterSet.Config as cms

hltPhase2PixelRecHitsExtendedSoA = cms.EDProducer('SiPixelRecHitExtendedAlpaka@alpaka',
    pixelRecHitsSoA = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    trackerRecHitsSoA = cms.InputTag('hltPhase2OtRecHitsSoA'),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
      backend = cms.untracked.string('')
    )
  )
