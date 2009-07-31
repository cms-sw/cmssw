import FWCore.ParameterSet.Config as cms

pfAllElectrons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("pfNoMuon"),
    pdgId = cms.vint32(11,-11)
)



