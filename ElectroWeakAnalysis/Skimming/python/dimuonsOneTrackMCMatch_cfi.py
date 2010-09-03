import FWCore.ParameterSet.Config as cms

dimuonsOneTrackMCMatch = cms.EDProducer("MCTruthCompositeMatcherNew",
    src = cms.InputTag("dimuonsOneTrack"),
    #
    # comment PAT match because works only for layer-0 muons
    #
    #  VInputTag matchMaps = { muonMatch, goodTrackMCMatch }
    matchPDGId = cms.vint32(23),
    matchMaps = cms.VInputTag(cms.InputTag("goodMuonMCMatch"), cms.InputTag("goodTrackMCMatch"))
)


