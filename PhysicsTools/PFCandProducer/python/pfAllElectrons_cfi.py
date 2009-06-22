import FWCore.ParameterSet.Config as cms

pfAllElectrons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("noPileUp"),
    pdgId = cms.vint32(11,-11)
)



