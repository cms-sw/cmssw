import FWCore.ParameterSet.Config as cms

cmgDiMuSel = cms.EDFilter(
    "PATCompositeCandidateSelector",
    src = cms.InputTag("cmgDiMu"),
    cut = cms.string("")
    )
