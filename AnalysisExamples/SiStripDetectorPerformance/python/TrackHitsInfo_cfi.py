import FWCore.ParameterSet.Config as cms

# TrackHitsInfo Module default configuration
modTrackHitsInfo = cms.EDFilter("TrackHitsInfo",
    # Tracks Labels
    oTrack = cms.untracked.InputTag("rsWithMaterialTracksTIFTIBTOB")
)


