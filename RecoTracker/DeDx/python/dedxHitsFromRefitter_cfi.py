import FWCore.ParameterSet.Config as cms

dedxHitsFromRefitter = cms.EDProducer("DeDxHitsProducer",
    refittedTracks = cms.InputTag("TrackRefitter"),
    tracks = cms.InputTag("ctfWithMaterialTracks"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter")
)


