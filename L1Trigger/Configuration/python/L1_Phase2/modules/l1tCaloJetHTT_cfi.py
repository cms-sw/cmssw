import FWCore.ParameterSet.Config as cms

l1tCaloJetHTT = cms.EDProducer("L1CaloJetHTTProducer",
    BXVCaloJetsInputTag = cms.InputTag("l1tCaloJet","L1CaloJetCollectionBXV"),
    EtaMax = cms.double(2.4),
    PtMin = cms.double(30.0),
    debug = cms.bool(False),
    genJets = cms.InputTag("ak4GenJetsNoNu"),
    use_gen_jets = cms.bool(False)
)
