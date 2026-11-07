import FWCore.ParameterSet.Config as cms

hltPhase2OtCAHitsSoA = cms.EDProducer('Phase2OTCAHitsAlpaka@alpaka',
  otClusters = cms.InputTag('hltSiPhase2Clusters'),
  pixelRecHitSoASource = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
  beamSpot = cms.InputTag('hltOnlineBeamSpot'),
  moduleParams = cms.ESInputTag('', 'HLT'),
  mightGet = cms.optional.untracked.vstring,
  alpaka = cms.untracked.PSet(
    backend = cms.untracked.string('')
  )
)
