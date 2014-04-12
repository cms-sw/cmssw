import FWCore.ParameterSet.Config as cms

qcdSingleJetFilterMed = cms.EDFilter("QCDSingleJetFilter",
    TriggerJetCollectionB = cms.InputTag("fastjet6CaloJets"),
    MinPt = cms.double(500.0),
    TriggerJetCollectionA = cms.InputTag("midPointCone7CaloJets")
)


