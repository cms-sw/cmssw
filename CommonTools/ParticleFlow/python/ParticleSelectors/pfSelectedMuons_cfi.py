import FWCore.ParameterSet.Config as cms

pfSelectedMuons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("pfMuonsFromVertex"),
    cut = cms.string("pt>5")
)




