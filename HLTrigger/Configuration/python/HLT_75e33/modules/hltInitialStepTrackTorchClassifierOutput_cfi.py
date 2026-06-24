import FWCore.ParameterSet.Config as cms

hltInitialStepTrackTorchClassifierOutput = cms.EDProducer("TrackTorchClassifierFromSoA",
    src = cms.InputTag("hltInitialStepTracks"),
    scores = cms.InputTag("hltInitialStepTrackTorchClassifier"),
    features = cms.InputTag("hltInitialStepTrackFeatureExtractor"),
    copyTrajectories = cms.bool(False),
    minScore = cms.double(0.377),
    dxyThreshold = cms.double(0.5),
    highDxyMinScore = cms.double(0.267)
)

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toModify(hltInitialStepTrackTorchClassifierOutput, copyTrajectories = True)
