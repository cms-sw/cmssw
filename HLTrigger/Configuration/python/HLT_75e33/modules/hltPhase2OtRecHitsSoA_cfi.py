import FWCore.ParameterSet.Config as cms

hltPhase2OtRecHitsSoA = cms.EDProducer('Phase2OTRecHitsSoAConverter@alpaka',
  pixelRecHitSoASource = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
  otRecHitSource = cms.InputTag('hltSiPhase2RecHits'),
  beamSpot = cms.InputTag('hltOnlineBeamSpot'),
  mightGet = cms.optional.untracked.vstring,
  alpaka = cms.untracked.PSet(
    backend = cms.untracked.string('')
  )
)
