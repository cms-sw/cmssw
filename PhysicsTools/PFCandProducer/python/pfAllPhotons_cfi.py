import FWCore.ParameterSet.Config as cms

pfAllPhotons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("particleFlow"),
    pdgId = cms.vint32(22)
)



