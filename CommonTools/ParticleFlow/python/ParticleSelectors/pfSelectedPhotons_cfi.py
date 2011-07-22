import FWCore.ParameterSet.Config as cms

pfSelectedPhotons = cms.EDFilter(
    "GenericPFCandidateSelector",
    src = cms.InputTag("pfAllPhotons"),
    cut = cms.string("mva_nothing_gamma>0")
)




