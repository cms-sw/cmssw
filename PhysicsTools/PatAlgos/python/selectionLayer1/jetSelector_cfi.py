import FWCore.ParameterSet.Config as cms

selectedLayer1Jets = cms.EDFilter("PATJetSelector",
    src = cms.InputTag("allLayer1Jets"),
    cut = cms.string('et > 15. & abs(eta) < 2.4 & nConstituents > 0')
)


