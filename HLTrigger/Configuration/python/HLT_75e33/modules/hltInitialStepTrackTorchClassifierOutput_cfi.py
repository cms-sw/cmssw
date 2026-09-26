import FWCore.ParameterSet.Config as cms

hltInitialStepTrackTorchClassifierOutput = cms.EDProducer("TrackTorchClassifierFromSoA",
    src = cms.InputTag("hltInitialStepTracks"),
    scores = cms.InputTag("hltInitialStepTrackTorchClassifier"),
    features = cms.InputTag("hltInitialStepTrackFeatureExtractor"),
    qualityCutsPrompt = cms.vdouble(0.377, 0.377, 0.377), #all 99.5%
    dxyThreshold = cms.double(0.5),
    qualityCutsDisplaced = cms.vdouble(0.267, 0.267, 0.267) #all 99.5%
)

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toModify(hltInitialStepTrackTorchClassifierOutput, copyTrajectories = True)
