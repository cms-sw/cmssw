import FWCore.ParameterSet.Config as cms

QCDSingleJetFilter = cms.EDFilter("QCDSingleJetFilter",
    PreScale = cms.double(1.0),
    TriggerJetCollectionB = cms.InputTag("Fastjet6CaloJets"),
    MinPt = cms.double(3000.0),
    TriggerJetCollectionA = cms.InputTag("midPointCone7CaloJets")
)


