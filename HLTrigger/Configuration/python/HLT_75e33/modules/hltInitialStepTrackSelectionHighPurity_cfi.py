import FWCore.ParameterSet.Config as cms

hltInitialStepTrackSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltInitialStepTrackTorchClassifierOutput","MVAValues"),
    originalQualVals = cms.InputTag("hltInitialStepTrackTorchClassifierOutput","QualityMasks"),
    originalSource = cms.InputTag("hltInitialStepTracks")
)

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toModify(hltInitialStepTrackSelectionHighPurity, copyTrajectories = True)

from Configuration.ProcessModifiers.trackCutClassifier_cff import trackCutClassifier
trackCutClassifier.toModify(hltInitialStepTrackSelectionHighPurity, originalSource = "hltInitialStepTracks")
