import FWCore.ParameterSet.Config as cms

pfAllPhotons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("pfNoPileUp"),
    pdgId = cms.vint32(22)
)



