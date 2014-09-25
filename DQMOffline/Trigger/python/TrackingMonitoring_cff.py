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
iter0TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter0'
iter0TracksMonitoringHLT.TrackProducer    = 'hltIter0PFlowCtfWithMaterialTracks'
iter0TracksMonitoringHLT.allTrackProducer = 'hltIter0PFlowCtfWithMaterialTracks'
iter0TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter0TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter0TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter0HPTracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter0HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter0HP'
iter0HPTracksMonitoringHLT.TrackProducer    = 'hltIter0PFlowTrackSelectionHighPurity'
iter0HPTracksMonitoringHLT.allTrackProducer = 'hltIter0PFlowTrackSelectionHighPurity'
iter0HPTracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter0HPTracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter0HPTracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter1TracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter1TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter1'
iter1TracksMonitoringHLT.TrackProducer    = 'hltIter1PFlowCtfWithMaterialTracks'
iter1TracksMonitoringHLT.allTrackProducer = 'hltIter1PFlowCtfWithMaterialTracks'
iter1TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter1TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter1TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter1HPTracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter1HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter1HP'
iter1HPTracksMonitoringHLT.TrackProducer    = 'hltIter1PFlowTrackSelectionHighPurity'
iter1HPTracksMonitoringHLT.allTrackProducer = 'hltIter1PFlowTrackSelectionHighPurity'
iter1HPTracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter1HPTracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter1HPTracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter2TracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter2TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2'
iter2TracksMonitoringHLT.TrackProducer    = 'hltIter2PFlowCtfWithMaterialTracks'
iter2TracksMonitoringHLT.allTrackProducer = 'hltIter2PFlowCtfWithMaterialTracks'
iter2TracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter2TracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter2TracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iter2HPTracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iter2HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2HP'
iter2HPTracksMonitoringHLT.TrackProducer    = 'hltIter2PFlowTrackSelectionHighPurity'
iter2HPTracksMonitoringHLT.allTrackProducer = 'hltIter2PFlowTrackSelectionHighPurity'
iter2HPTracksMonitoringHLT.doAllPlots       = cms.bool(False)
iter2HPTracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iter2HPTracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

iterHLTTracksMonitoringHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
iterHLTTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2Merged'
iterHLTTracksMonitoringHLT.TrackProducer    = 'hltIter2Merged'
iterHLTTracksMonitoringHLT.allTrackProducer = 'hltIter2Merged'
iterHLTTracksMonitoringHLT.doAllPlots       = cms.bool(False)
iterHLTTracksMonitoringHLT.doLumiAnalysis   = cms.bool(False)     
iterHLTTracksMonitoringHLT.doProfilesVsLS   = cms.bool(False)

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
    + iter0HPTracksMonitoringHLT
#    + iter1HPTracksMonitoringHLT
#    + iter2HPTracksMonitoringHLT
    + iterHLTTracksMonitoringHLT
)    

trackingMonitorHLTall = cms.Sequence(
    pixelTracksMonitoringHLT
    + iter0TracksMonitoringHLT
    + iter2HPTracksMonitoringHLT
    + iter1TracksMonitoringHLT
    + iter1HPTracksMonitoringHLT
    + iter2TracksMonitoringHLT
    + iter2HPTracksMonitoringHLT
    + iterHLTTracksMonitoringHLT
#    + iter3TracksMonitoringHLT
#    + iter4TracksMonitoringHLT
)    

