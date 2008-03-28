import FWCore.ParameterSet.Config as cms

goodZToMuMuOneTrackFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("goodZToMuMuOneTrack"),
    minNumber = cms.uint32(1)
)


