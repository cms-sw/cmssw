import FWCore.ParameterSet.Config as cms

goodZToMuMuOneStandAloneMuonTrackFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuonTrack"),
    minNumber = cms.uint32(1)
)


