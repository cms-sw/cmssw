import FWCore.ParameterSet.Config as cms

trackinfo = cms.EDProducer("TrackInfoProducer",
    combinedState = cms.string('combinedState'),
    forwardPredictedState = cms.string(''),
    updatedState = cms.string('updatedState'),
    cosmicTracks = cms.InputTag("ctfWithMaterialTracks"),
    backwardPredictedState = cms.string(''),
    rechits = cms.InputTag("ctfWithMaterialTracks")
)


