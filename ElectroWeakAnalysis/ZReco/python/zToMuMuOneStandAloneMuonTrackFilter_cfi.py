import FWCore.ParameterSet.Config as cms

zToMuMuOneStandAloneMuonTrackFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("zToMuMuOneStandAloneMuonTrack"),
    minNumber = cms.uint32(1)
)


