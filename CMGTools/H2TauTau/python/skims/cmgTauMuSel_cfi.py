import FWCore.ParameterSet.Config as cms

cmgTauMuSel = cms.EDFilter(
    "PATCompositeCandidateSelector",
    src = cms.InputTag("cmgTauMu"),
    cut = cms.string("")
    )
