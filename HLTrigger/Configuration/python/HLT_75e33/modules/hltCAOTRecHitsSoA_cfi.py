import FWCore.ParameterSet.Config as cms

hltCAOTRecHitsSoA = cms.EDProducer('CAOTRecHitsSoAConverter@alpaka',
  otRecHitSource = cms.InputTag('hltSiPhase2RecHits'),
  beamSpot = cms.InputTag('hltOnlineBeamSpot'),
  psOnly = cms.bool(False),
  mightGet = cms.optional.untracked.vstring,
  alpaka = cms.untracked.PSet(
    backend = cms.untracked.string('')
  )
)
