import FWCore.ParameterSet.Config as cms

dtSegmentSelectionResiduals = cms.PSet(
    checkNoisyChannels = cms.bool(False),
    minHitsPhi = cms.int32(8),
    minHitsZ = cms.int32(4),
    maxChi2 = cms.double(1000.0),
    maxAnglePhi = cms.double(25.),
    maxAngleZ = cms.double(999.)
)

dtResidualCalibration = cms.EDAnalyzer("DTResidualCalibration",
    # Segment selection
    dtSegmentSelectionResiduals,
    segment4DLabel = cms.InputTag('dt4DSegments'),
    rootBaseDir = cms.untracked.string('DTResiduals'),
    rootFileName = cms.untracked.string('residuals.root')
)
