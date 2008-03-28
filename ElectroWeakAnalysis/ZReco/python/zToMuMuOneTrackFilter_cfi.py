import FWCore.ParameterSet.Config as cms

zToMuMuOneTrackFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("zToMuMuOneTrack"),
    minNumber = cms.uint32(1)
)


