import FWCore.ParameterSet.Config as cms

ecalDetailedTimeRecHit = cms.EDProducer("EcalDetailedTimeRecHitProducer",
    EBDetailedTimeRecHitCollection = cms.string('EcalRecHitsEB'),
    EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EBTimeDigiCollection = cms.InputTag("mix","EBTimeDigi"),
    EBTimeLayer = cms.int32(7),
    EEDetailedTimeRecHitCollection = cms.string('EcalRecHitsEE'),
    EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EETimeDigiCollection = cms.InputTag("mix","EETimeDigi"),
    EETimeLayer = cms.int32(3),
    correctForVertexZPosition = cms.bool(False),
    recoVertex = cms.InputTag("offlinePrimaryVerticesWithBS"),
    simVertex = cms.InputTag("g4SimHits"),
    useMCTruthVertex = cms.bool(False)
)
