import FWCore.ParameterSet.Config as cms

pfMuonsPtGt5 = cms.EDProducer("PtMinPFCandidateSelector",
    src = cms.InputTag("pfAllMuons"),
    ptMin = cms.double(5.0)
)




