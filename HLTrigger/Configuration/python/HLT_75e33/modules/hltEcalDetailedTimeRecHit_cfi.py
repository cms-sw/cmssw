import FWCore.ParameterSet.Config as cms

hltEcalDetailedTimeRecHit = cms.EDProducer("EcalDetailedTimeRecHitProducer",
    EBDetailedTimeRecHitCollection = cms.string('EcalRecHitsEB'),
    EBRecHitCollection = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
    EBTimeDigiCollection = cms.InputTag("mix","EBTimeDigi"),
    EBTimeLayer = cms.int32(7),
    EEDetailedTimeRecHitCollection = cms.string('EcalRecHitsEE'),
    EERecHitCollection = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
    EETimeDigiCollection = cms.InputTag("mix","EETimeDigi"),
    EETimeLayer = cms.int32(3),
    correctForVertexZPosition = cms.bool(False),
    simVertex = cms.InputTag("g4SimHits"),
    useMCTruthVertex = cms.bool(False)
)
