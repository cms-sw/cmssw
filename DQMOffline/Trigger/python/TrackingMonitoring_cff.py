import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
trackingMonHLT = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
trackingMonHLT.beamSpot         = cms.InputTag("hltOnlineBeamSpot")
trackingMonHLT.primaryVertex    = cms.InputTag("hltPixelVertices")
trackingMonHLT.doAllPlots       = cms.bool(False)
trackingMonHLT.doLumiAnalysis   = cms.bool(False)     
trackingMonHLT.doProfilesVsLS   = cms.bool(False)
trackingMonHLT.pvNDOF           = cms.int32(1)

pixelTracksMonitoringHLT = trackingMonHLT.clone()
pixelTracksMonitoringHLT.FolderName       = 'HLT/Tracking/pixelTracks'
pixelTracksMonitoringHLT.TrackProducer    = 'hltPixelTracks'
pixelTracksMonitoringHLT.allTrackProducer = 'hltPixelTracks'

iter0TracksMonitoringHLT = trackingMonHLT.clone()
iter0TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter0'
iter0TracksMonitoringHLT.TrackProducer    = 'hltIter0PFlowCtfWithMaterialTracks'
iter0TracksMonitoringHLT.allTrackProducer = 'hltIter0PFlowCtfWithMaterialTracks'

iter0HPTracksMonitoringHLT = trackingMonHLT.clone()
iter0HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter0HP'
iter0HPTracksMonitoringHLT.TrackProducer    = 'hltIter0PFlowTrackSelectionHighPurity'
iter0HPTracksMonitoringHLT.allTrackProducer = 'hltIter0PFlowTrackSelectionHighPurity'

iter1TracksMonitoringHLT = trackingMonHLT.clone()
iter1TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter1'
iter1TracksMonitoringHLT.TrackProducer    = 'hltIter1PFlowCtfWithMaterialTracks'
iter1TracksMonitoringHLT.allTrackProducer = 'hltIter1PFlowCtfWithMaterialTracks'

iter1HPTracksMonitoringHLT = trackingMonHLT.clone()
iter1HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter1HP'
iter1HPTracksMonitoringHLT.TrackProducer    = 'hltIter1PFlowTrackSelectionHighPurity'
iter1HPTracksMonitoringHLT.allTrackProducer = 'hltIter1PFlowTrackSelectionHighPurity'

iter2TracksMonitoringHLT = trackingMonHLT.clone()
iter2TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2'
iter2TracksMonitoringHLT.TrackProducer    = 'hltIter2PFlowCtfWithMaterialTracks'
iter2TracksMonitoringHLT.allTrackProducer = 'hltIter2PFlowCtfWithMaterialTracks'

iter2HPTracksMonitoringHLT = trackingMonHLT.clone()
iter2HPTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2HP'
iter2HPTracksMonitoringHLT.TrackProducer    = 'hltIter2PFlowTrackSelectionHighPurity'
iter2HPTracksMonitoringHLT.allTrackProducer = 'hltIter2PFlowTrackSelectionHighPurity'

iterHLTTracksMonitoringHLT = trackingMonHLT.clone()
iterHLTTracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter2Merged'
iterHLTTracksMonitoringHLT.TrackProducer    = 'hltIter2Merged'
iterHLTTracksMonitoringHLT.allTrackProducer = 'hltIter2Merged'

iter3TracksMonitoringHLT = trackingMonHLT.clone()
iter3TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter3Merged'
iter3TracksMonitoringHLT.TrackProducer    = 'hltIter3Merged'
iter3TracksMonitoringHLT.allTrackProducer = 'hltIter3Merged'

iter4TracksMonitoringHLT = trackingMonHLT.clone()
iter4TracksMonitoringHLT.FolderName       = 'HLT/Tracking/iter4Merged'
iter4TracksMonitoringHLT.TrackProducer    = 'hltIter4Merged'
iter4TracksMonitoringHLT.allTrackProducer = 'hltIter4Merged'

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

