import FWCore.ParameterSet.Config as cms

L1CaloJetHTTProducer = cms.EDProducer("L1CaloJetHTTProducer",
    debug = cms.bool(False),
    PtMin = cms.double(30.0),
    EtaMax = cms.double(2.4),
    BXVCaloJetsInputTag = cms.InputTag("L1CaloJetProducer","L1CaloJetCollectionBXV")
)
