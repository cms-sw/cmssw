import FWCore.ParameterSet.Config as cms

#HLTPixelIsolTrackFilter configuration
isolPixelTrackFilter = cms.EDFilter("HLTPixelIsolTrackFilter",
    MaxPtNearby = cms.double(2.0),
    MinEnergyTrack = cms.double(15.0),
    MinPtTrack = cms.double(20.0),
    MaxEtaTrack = cms.double(1.9),
    candTag = cms.InputTag("isolPixelTrackProd"),
    filterTrackEnergy = cms.bool(True),
    NMaxTrackCandidates=cms.int32(10),
    DropMultiL2Event = cms.bool(False) 
)


