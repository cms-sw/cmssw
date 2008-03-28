import FWCore.ParameterSet.Config as cms

zToEEOneTrackFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("zToEEOneTrack"),
    minNumber = cms.uint32(1)
)


