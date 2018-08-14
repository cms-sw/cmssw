import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
pixelTracksMonitoring = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
pixelTracksMonitoring.FolderName                = 'Tracking/PixelTrackParameters'
pixelTracksMonitoring.TrackProducer             = 'pixelTracks'
pixelTracksMonitoring.allTrackProducer          = 'pixelTracks'
pixelTracksMonitoring.beamSpot                  = 'offlineBeamSpot'
pixelTracksMonitoring.primaryVertex             = 'pixelVertices'
pixelTracksMonitoring.pvNDOF                    = 1
pixelTracksMonitoring.doAllPlots                = True
pixelTracksMonitoring.doLumiAnalysis            = True
pixelTracksMonitoring.doProfilesVsLS            = True
pixelTracksMonitoring.doDCAPlots                = True
pixelTracksMonitoring.doProfilesVsLS            = True
pixelTracksMonitoring.doPlotsVsGoodPVtx         = True
pixelTracksMonitoring.doEffFromHitPatternVsPU   = False
pixelTracksMonitoring.doEffFromHitPatternVsBX   = False
pixelTracksMonitoring.doEffFromHitPatternVsLUMI = False
pixelTracksMonitoring.doPlotsVsGoodPVtx         = True
pixelTracksMonitoring.doPlotsVsLUMI             = True
pixelTracksMonitoring.doPlotsVsBX               = True

