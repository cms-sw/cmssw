import FWCore.ParameterSet.Config as cms
from L1Trigger.L1TTrackMatch.L1TrackerEtMissEmulatorProducer_cfi import L1TrackerEmuEtMiss
from L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi import L1TrackerEtMiss

L1TkMETAnalyser = cms.EDAnalyzer('L1TkMETAnalyser',
  TrackMETInputTag = cms.InputTag("L1TrackerEtMiss",L1TrackerEtMiss.L1MetCollectionName.value()),
  TrackMETEmuInputTag = cms.InputTag("L1TrackerEmuEtMiss",L1TrackerEmuEtMiss.L1MetCollectionName.value()),
  TrackMETHWInputTag = cms.InputTag("GTTOutputFileReader"),
  HW_Analysis = cms.bool(False)
)