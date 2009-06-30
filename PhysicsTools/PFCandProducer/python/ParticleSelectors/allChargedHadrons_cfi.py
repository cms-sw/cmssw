import FWCore.ParameterSet.Config as cms

allChargedHadrons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("noPileUp"),
    pdgId = cms.vint32(211,-211,321,-321,999211,2212,-2212)
)



