import FWCore.ParameterSet.Config as cms

pfAllNeutralHadrons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("particleFlow"),
    pdgId = cms.vint32(111,130,310,2112)
)



