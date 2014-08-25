import FWCore.ParameterSet.Config as cms

matchGenHFHadron = cms.EDProducer("GenHFHadronMatcher",
    genJets = cms.InputTag('ak5GenJets','','SIM'),   
    flavour = cms.int32(5),
    onlyJetClusteredHadrons = cms.bool(False),
    noBBbarResonances = cms.bool(True),
)



