import FWCore.ParameterSet.Config as cms

from CalibMuon.DTCalibration.dtSegmentSelection_cfi import dtSegmentSelection

dtResidualCalibration = cms.EDAnalyzer("DTResidualCalibration",
    # Segment selection
    dtSegmentSelection,
    histogramRange = cms.double(0.4),
    segment4DLabel = cms.InputTag('dt4DSegments'),
    rootBaseDir = cms.untracked.string('DTResiduals'),
    rootFileName = cms.untracked.string('residuals.root'),
    detailedAnalysis = cms.untracked.bool(False)
    #detailedAnalysis = cms.untracked.bool(True)
)
