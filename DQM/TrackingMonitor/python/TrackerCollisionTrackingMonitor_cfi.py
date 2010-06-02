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

TrackerCollisionTrackMon.doLumiAnalysis        = cms.bool(True)                       

# Number of Tracks per Event
TrackerCollisionTrackMon.TkSizeBin             = cms.int32(300)
TrackerCollisionTrackMon.TkSizeMax             = cms.double(299.5)                        
TrackerCollisionTrackMon.TkSizeMin             = cms.double(-0.5)

# chi2 dof
TrackerCollisionTrackMon.Chi2NDFBin            = cms.int32(100)
TrackerCollisionTrackMon.Chi2NDFMax            = cms.double(49.5)
TrackerCollisionTrackMon.Chi2NDFMin            = cms.double(-0.5)
                
