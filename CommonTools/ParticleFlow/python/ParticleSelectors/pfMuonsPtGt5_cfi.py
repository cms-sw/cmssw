import FWCore.ParameterSet.Config as cms

pfMuonsPtGt5 = cms.EDFilter("PtMinPFCandidateSelector",
    src = cms.InputTag("pfAllMuons"),
    ptMin = cms.double(5.0)
)




