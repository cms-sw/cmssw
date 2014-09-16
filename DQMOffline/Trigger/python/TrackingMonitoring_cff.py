import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
pixelTracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
pixelTracksMonitoringHLT.FolderName       = 'HLT/Tracking/pixelTracks'
pixelTracksMonitoringHLT.TrackProducer    = 'hltPixelTracks'
pixelTracksMonitoringHLT.allTrackProducer = 'hltPixelTracks'
pixelTracksMonitoringHLT.doAllPlots       = cms.bool(False)
pixelTracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
pixelTracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)


iter0TracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter0TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter0Merged'
iter0TracksMonitoringHLT.TrackProducer    = 'hltIter0Merged'
iter0TracksMonitoringHLT.allTrackProducer = 'hltIter0Merged'
iter0TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter0TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter0TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter1TracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter1TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter1Merged'
iter1TracksMonitoringHLT.TrackProducer    = 'hltIter1Merged'
iter1TracksMonitoringHLT.allTrackProducer = 'hltIter1Merged'
iter1TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter1TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter1TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter2TracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter2TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2Merged'
iter2TracksMonitoringHLT.TrackProducer    = 'hltIter2Merged'
iter2TracksMonitoringHLT.allTrackProducer = 'hltIter2Merged'
iter2TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter2TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter2TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter3TracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter3TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter3Merged'
iter3TracksMonitoringHLT.TrackProducer    = 'hltIter3Merged'
iter3TracksMonitoringHLT.allTrackProducer = 'hltIter3Merged'
iter3TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter3TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter3TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter4TracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter4TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter4Merged'
iter4TracksMonitoringHLT.TrackProducer    = 'hltIter4Merged'
iter4TracksMonitoringHLT.allTrackProducer = 'hltIter4Merged'
iter4TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter4TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter4TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

trackingMonitorHLT = cms.Sequence(
    pixelTracksMonitoringHLT
    + iter4TracksMonitoringHLT
)    

trackingMonitorHLTall = cms.Sequence(
    pixelTracksMonitoringHLT
    + iter0TracksMonitoringHLT
    + iter1TracksMonitoringHLT
    + iter2TracksMonitoringHLT
    + iter3TracksMonitoringHLT
    + iter4TracksMonitoringHLT
)    

