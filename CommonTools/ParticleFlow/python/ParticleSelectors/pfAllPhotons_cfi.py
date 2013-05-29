import FWCore.ParameterSet.Config as cms

pfAllPhotons = cms.EDFilter("PdgIdPFCandidateSelector",
    src = cms.InputTag("pfNoPileUpIso"),
    pdgId = cms.vint32(22)
)



