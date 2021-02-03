import FWCore.ParameterSet.Config as cms

JetIDParams = cms.PSet(
    ebRecHitsColl = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    eeRecHitsColl = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    hbheRecHitsColl = cms.InputTag("hbhereco"),
    hfRecHitsColl = cms.InputTag("hfreco"),
    hoRecHitsColl = cms.InputTag("horeco"),
    rpcRecHits = cms.InputTag("rpcRecHits"),
    useRecHits = cms.bool(True)
)