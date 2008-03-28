import FWCore.ParameterSet.Config as cms

#HLTPixelIsolTrackFilter configuration
isolPixelTrackFilter = cms.EDFilter("HLTPixelIsolTrackFilter",
    MaxPtNearby = cms.double(2.0),
    candTag = cms.InputTag("isolPixelTrackProd"),
    MaxEtaTrack = cms.double(1.3),
    MinPtTrack = cms.double(20.0)
)


