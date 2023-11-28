import FWCore.ParameterSet.Config as cms

L1TkMETAnalyser = cms.EDAnalyzer('L1TkMETAnalyser',
  TrackMETInputTag = cms.InputTag("l1tTrackerEtMiss","L1TrackerEtMiss"),
  TrackMETEmuInputTag = cms.InputTag("l1tTrackerEmuEtMiss","L1TrackerEmuEtMiss"),
  TrackMETHWInputTag = cms.InputTag("GTTOutputFileReader"),
  HW_Analysis = cms.bool(False)
)
