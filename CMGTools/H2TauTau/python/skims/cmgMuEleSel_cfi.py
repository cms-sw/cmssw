import FWCore.ParameterSet.Config as cms

cmgMuEleSel = cms.EDFilter(
    "PATCompositeCandidateSelector",
    src = cms.InputTag("cmgMuEle"),
    cut = cms.string("")
    )


