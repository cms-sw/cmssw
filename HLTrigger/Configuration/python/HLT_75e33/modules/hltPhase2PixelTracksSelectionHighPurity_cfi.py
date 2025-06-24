import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracksSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("hltPhase2PixelTracks")
)
