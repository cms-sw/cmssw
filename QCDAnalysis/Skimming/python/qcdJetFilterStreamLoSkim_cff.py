import FWCore.ParameterSet.Config as cms

qcdSingleJetFilterLo = cms.EDFilter("QCDSingleJetFilter",
    TriggerJetCollectionB = cms.InputTag("fastjet6CaloJets"),
    MinPt = cms.double(140.0),
    TriggerJetCollectionA = cms.InputTag("midPointCone7CaloJets")
)


