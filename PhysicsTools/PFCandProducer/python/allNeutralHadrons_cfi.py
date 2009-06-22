import FWCore.ParameterSet.Config as cms

allNeutralHadrons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("noPileUp"),
    pdgId = cms.vint32(111,130,310,2112)
)



