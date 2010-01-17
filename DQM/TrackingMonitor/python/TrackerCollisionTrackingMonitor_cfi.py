import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackerCollisionTrackMon = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()

# Update specific parameters

TrackerCollisionTrackMon.TrackProducer         = cms.InputTag("generalTracks")
TrackerCollisionTrackMon.SeedProducer          = cms.InputTag("newSeedFromTriplets")
TrackerCollisionTrackMon.TCProducer            = cms.InputTag("newTrackCandidateMaker")
TrackerCollisionTrackMon.beamSpot              = cms.InputTag("offlineBeamSpot")

TrackerCollisionTrackMon.AlgoName              = cms.string('GenTk')
TrackerCollisionTrackMon.Quality               = cms.string('')
TrackerCollisionTrackMon.FolderName            = cms.string('Tracking/GlobalParameters')
TrackerCollisionTrackMon.BSFolderName          = cms.string('Tracking/BeamSpotParameters')

TrackerCollisionTrackMon.MeasurementState      = cms.string('ImpactPoint')

TrackerCollisionTrackMon.doTrackerSpecific     = cms.bool(True)
TrackerCollisionTrackMon.doAllPlots            = cms.bool(True)
TrackerCollisionTrackMon.doBeamSpotPlots       = cms.bool(True)
TrackerCollisionTrackMon.doSeedParameterHistos = cms.bool(True)
