import FWCore.ParameterSet.Config as cms

from CalibMuon.DTCalibration.dtSegmentSelection_cfi import dtSegmentSelection

dtVDriftSegmentCalibration = cms.EDAnalyzer("DTVDriftSegmentCalibration",
    # Segment selection
    dtSegmentSelection,
    recHits4DLabel = cms.InputTag('dt4DSegments'),
    rootFileName = cms.untracked.string('DTVDriftHistos.root'),
    # Choose the chamber you want to calibrate (default = "All"), specify the chosen chamber
    # in the format "wheel station sector" (i.e. "-1 3 10")
    calibChamber = cms.untracked.string('All')
)
