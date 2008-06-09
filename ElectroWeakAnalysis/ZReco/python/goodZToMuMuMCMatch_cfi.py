import FWCore.ParameterSet.Config as cms

goodZToMuMuMCMatch = cms.EDFilter("MCTruthCompositeMatcherNew",
    src = cms.InputTag("goodZToMuMu"),
    matchPDGId = cms.vint32(),
    matchMaps = cms.VInputTag(cms.InputTag("goodMuonMCMatch"))
)


