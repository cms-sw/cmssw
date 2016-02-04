import FWCore.ParameterSet.Config as cms

qcdSingleJetFilter = cms.EDFilter("QCDSingleJetFilter",
    TriggerJetCollectionB = cms.InputTag("fastjet6CaloJets"),
    MinPt = cms.double(3000.0),
    TriggerJetCollectionA = cms.InputTag("midPointCone7CaloJets")
)


