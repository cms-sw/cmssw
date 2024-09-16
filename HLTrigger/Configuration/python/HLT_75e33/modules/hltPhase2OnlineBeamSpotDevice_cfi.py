import FWCore.ParameterSet.Config as cms

hltPhase2OnlineBeamSpotDevice = cms.EDProducer('BeamSpotDeviceProducer@alpaka',
    src = cms.InputTag('hltOnlineBeamSpot'),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
