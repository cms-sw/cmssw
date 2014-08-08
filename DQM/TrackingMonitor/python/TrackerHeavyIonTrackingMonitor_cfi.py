import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackerHeavyIonTrackMon = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()

# Update specific parameters

TrackerHeavyIonTrackMon.TrackProducer         = cms.InputTag("hiGeneralTracks")
TrackerHeavyIonTrackMon.SeedProducer          = cms.InputTag("hiPixelTrackSeeds")
TrackerHeavyIonTrackMon.TCProducer            = cms.InputTag("hiPrimTrackCandidates")
TrackerHeavyIonTrackMon.beamSpot              = cms.InputTag("offlineBeamSpot")
TrackerHeavyIonTrackMon.primaryVertex         = cms.InputTag('hiSelectedVertex')


TrackerHeavyIonTrackMon.doHIPlots             = cms.bool(True)


TrackerHeavyIonTrackMon.AlgoName              = cms.string('HeavyIonTk')
TrackerHeavyIonTrackMon.Quality               = cms.string('')
TrackerHeavyIonTrackMon.FolderName            = cms.string('Tracking/GlobalParameters')
TrackerHeavyIonTrackMon.BSFolderName          = cms.string('Tracking/BeamSpotParameters')

TrackerHeavyIonTrackMon.MeasurementState      = cms.string('ImpactPoint')

TrackerHeavyIonTrackMon.doTrackerSpecific     = cms.bool(True)
TrackerHeavyIonTrackMon.doAllPlots            = cms.bool(True)
TrackerHeavyIonTrackMon.doBeamSpotPlots       = cms.bool(True)
TrackerHeavyIonTrackMon.doSeedParameterHistos = cms.bool(True)

TrackerHeavyIonTrackMon.doLumiAnalysis        = cms.bool(True)                       

# Number of Tracks per Event
TrackerHeavyIonTrackMon.TkSizeBin             = cms.int32(600)
TrackerHeavyIonTrackMon.TkSizeMax             = cms.double(1799.5)                        
TrackerHeavyIonTrackMon.TkSizeMin             = cms.double(-0.5)

# chi2 dof
TrackerHeavyIonTrackMon.Chi2NDFBin            = cms.int32(100)
TrackerHeavyIonTrackMon.Chi2NDFMax            = cms.double(49.5)
TrackerHeavyIonTrackMon.Chi2NDFMin            = cms.double(-0.5)
                
