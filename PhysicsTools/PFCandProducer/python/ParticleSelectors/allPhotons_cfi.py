import FWCore.ParameterSet.Config as cms

allPhotons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("noPileUp"),
    pdgId = cms.vint32(22)
)



