import FWCore.ParameterSet.Config as cms
from L1Trigger.L1TTrackMatch.L1TrackerEtMissEmulatorProducer_cfi import l1tTrackerEmuEtMiss
from L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi import l1tTrackerEtMiss

L1TkMETAnalyser = cms.EDAnalyzer('L1TkMETAnalyser',
  TrackMETInputTag = cms.InputTag("l1tTrackerEtMiss",l1tTrackerEtMiss.L1MetCollectionName.value()),
  TrackMETEmuInputTag = cms.InputTag("l1tTrackerEmuEtMiss",l1tTrackerEmuEtMiss.L1MetCollectionName.value()),
  TrackMETHWInputTag = cms.InputTag("GTTOutputFileReader"),
  HW_Analysis = cms.bool(False)
)
