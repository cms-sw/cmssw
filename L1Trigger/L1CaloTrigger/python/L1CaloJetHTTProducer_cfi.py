import FWCore.ParameterSet.Config as cms

l1tCaloJetHTTProducer = cms.EDProducer("L1CaloJetHTTProducer",
    EtaMax = cms.double(2.4),
    PtMin = cms.double(30.0),
    BXVCaloJetsInputTag = cms.InputTag("l1tCaloJetProducer","L1CaloJetCollectionBXV"),
    genJets = cms.InputTag("ak4GenJetsNoNu"),
    debug = cms.bool(False),
    use_gen_jets = cms.bool(False),
)
