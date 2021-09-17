import FWCore.ParameterSet.Config as cms

L1CaloJetHTTProducer = cms.EDProducer("L1CaloJetHTTProducer",
    EtaMax = cms.double(2.4),
    PtMin = cms.double(30.0),
    BXVCaloJetsInputTag = cms.InputTag("L1CaloJetProducer","L1CaloJetCollectionBXV"),
    genJets = cms.InputTag("ak4GenJetsNoNu", "", "HLT"),
    debug = cms.bool(False),
    use_gen_jets = cms.bool(False),
)
