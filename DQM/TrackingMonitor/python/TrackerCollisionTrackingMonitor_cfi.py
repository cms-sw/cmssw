import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackerCollisionTrackMon = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()

# Update specific parameters

TrackerCollisionTrackMon.doGoodTrackPlots      = cms.bool(True)

TrackerCollisionTrackMon.TrackProducer         = cms.InputTag("generalTracks")
TrackerCollisionTrackMon.SeedProducer          = cms.InputTag("initialStepSeeds")
TrackerCollisionTrackMon.TCProducer            = cms.InputTag("initialStepTrackCandidates")
TrackerCollisionTrackMon.beamSpot              = cms.InputTag("offlineBeamSpot")

TrackerCollisionTrackMon.AlgoName              = cms.string('GenTk')
TrackerCollisionTrackMon.Quality               = cms.string('')
TrackerCollisionTrackMon.FolderName            = cms.string('Tracking/GlobalParameters')
TrackerCollisionTrackMon.BSFolderName          = cms.string('Tracking/BeamSpotParameters')

TrackerCollisionTrackMon.MeasurementState      = cms.string('ImpactPoint')

TrackerCollisionTrackMon.doTrackerSpecific     = cms.bool(True)
TrackerCollisionTrackMon.doAllPlots            = cms.bool(False)
TrackerCollisionTrackMon.doHitPropertiesPlots  = cms.bool(True)
TrackerCollisionTrackMon.doGeneralPropertiesPlots  = cms.bool(True)
TrackerCollisionTrackMon.doBeamSpotPlots       = cms.bool(True)
TrackerCollisionTrackMon.doSeedParameterHistos = cms.bool(False)
TrackerCollisionTrackMon.doRecHitVsPhiVsEtaPerTrack = cms.bool(True)
TrackerCollisionTrackMon.doGoodTrackRecHitVsPhiVsEtaPerTrack = cms.bool(True)

TrackerCollisionTrackMon.doLumiAnalysis        = cms.bool(True)                       

TrackerCollisionTrackMon.doSeedNumberHisto = cms.bool(False)
TrackerCollisionTrackMon.doSeedETAHisto    = cms.bool(False)
TrackerCollisionTrackMon.doSeedVsClusterHisto = cms.bool(False)

# Number of Tracks per Event
TrackerCollisionTrackMon.TkSizeBin             = cms.int32(200)
TrackerCollisionTrackMon.TkSizeMax             = cms.double(999.5)                        
TrackerCollisionTrackMon.TkSizeMin             = cms.double(-0.5)

# chi2 dof
TrackerCollisionTrackMon.Chi2NDFBin            = cms.int32(50)
TrackerCollisionTrackMon.Chi2NDFMax            = cms.double(49.5)
TrackerCollisionTrackMon.Chi2NDFMin            = cms.double(-0.5)

# Number of seeds per Event
TrackerCollisionTrackMon.TkSeedSizeBin = cms.int32(100)
TrackerCollisionTrackMon.TkSeedSizeMax = cms.double(499.5)                        
TrackerCollisionTrackMon.TkSeedSizeMin = cms.double(-0.5)

# Number of Track Cadidates per Event
TrackerCollisionTrackMon.TCSizeBin = cms.int32(100)
TrackerCollisionTrackMon.TCSizeMax = cms.double(499.5)
TrackerCollisionTrackMon.TCSizeMin = cms.double(-0.5)
