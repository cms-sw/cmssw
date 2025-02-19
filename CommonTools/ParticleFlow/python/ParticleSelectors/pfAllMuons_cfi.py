import FWCore.ParameterSet.Config as cms

pfAllMuons = cms.EDFilter("PdgIdPFCandidateSelector",
    src = cms.InputTag("pfNoPileUp"),
    pdgId = cms.vint32( -13, 13)
)



