import FWCore.ParameterSet.Config as cms

qcdSingleJetFilterHi = cms.EDFilter("QCDSingleJetFilter",
    TriggerJetCollectionB = cms.InputTag("fastjet6CaloJets"),
    MinPt = cms.double(1000.0),
    TriggerJetCollectionA = cms.InputTag("midPointCone7CaloJets")
)


