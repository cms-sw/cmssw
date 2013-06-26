import FWCore.ParameterSet.Config as cms

pfPdgIdPFCandidateSelector = cms.EDFilter("PdgIdPFCandidateSelector",
  src = cms.InputTag("pfNoPileUpIso"),
  pdgId = cms.vint32(22,111,130,310,2112)
)

