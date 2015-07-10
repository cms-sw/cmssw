import FWCore.ParameterSet.Config as cms

cmgTauEleSel = cms.EDFilter(
    "PATCompositeCandidateSelector",
    src = cms.InputTag("cmgTauEle"),
    cut = cms.string("")
    )
