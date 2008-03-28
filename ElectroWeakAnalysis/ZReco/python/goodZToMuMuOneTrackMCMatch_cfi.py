import FWCore.ParameterSet.Config as cms

goodZToMuMuOneTrackMCMatch = cms.EDFilter("MCTruthCompositeMatcherNew",
    src = cms.InputTag("goodZToMuMuOneTrack"),
    matchPDGId = cms.vint32(23), ## Z

    matchMaps = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("goodTrackMCMatch"))
)


