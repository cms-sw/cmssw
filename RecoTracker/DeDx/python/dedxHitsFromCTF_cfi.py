import FWCore.ParameterSet.Config as cms

dedxHitsFromCTF = cms.EDProducer("DeDxHitsProducer",
    refittedTracks = cms.InputTag("ctfWithMaterialTracks"),
    tracks = cms.InputTag("ctfWithMaterialTracks"),
    trajectoryTrackAssociation = cms.InputTag("ctfWithMaterialTracks")
)


