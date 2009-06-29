import FWCore.ParameterSet.Config as cms

pfAllElectrons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("pfNoPileUp"),
    pdgId = cms.vint32(11,-11)
)



