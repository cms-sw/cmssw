import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackerCosmicTrackMon = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()

# Update specific parameters
TrackerCosmicTrackMon.SeedProducer          = cms.InputTag("combinedP5SeedsForCTF")
TrackerCosmicTrackMon.TCProducer            = cms.InputTag("ckfTrackCandidatesP5")
TrackerCosmicTrackMon.beamSpot              = cms.InputTag("offlineBeamSpot")              

TrackerCosmicTrackMon.MeasurementState      = cms.string('default')

TrackerCosmicTrackMon.doAllPlots            = cms.bool(False)
TrackerCosmicTrackMon.doHitPropertiesPlots     = cms.bool(True)
TrackerCosmicTrackMon.doGeneralPropertiesPlots = cms.bool(True)
TrackerCosmicTrackMon.doBeamSpotPlots       = cms.bool(False)
TrackerCosmicTrackMon.doSeedParameterHistos = cms.bool(False)

TrackerCosmicTrackMon.Chi2Max               = cms.double(500.0)

TrackerCosmicTrackMon.TkSizeBin             = cms.int32(25)
TrackerCosmicTrackMon.TkSizeMax             = cms.double(24.5)

TrackerCosmicTrackMon.TkSeedSizeBin         = cms.int32(20)
TrackerCosmicTrackMon.TkSeedSizeMax         = cms.double(19.5)

TrackerCosmicTrackMon.RecLayBin             = cms.int32(35)
TrackerCosmicTrackMon.RecLayMax             = cms.double(34.5)

TrackerCosmicTrackMon.TrackPtMax            = cms.double(30.0)
TrackerCosmicTrackMon.TrackPtMin            = cms.double(-0.5)

TrackerCosmicTrackMon.TrackPxMax            = cms.double(50.0)
TrackerCosmicTrackMon.TrackPxMin            = cms.double(-50.0)

TrackerCosmicTrackMon.TrackPyMax            = cms.double(50.0)
TrackerCosmicTrackMon.TrackPyMin            = cms.double(-50.0)

TrackerCosmicTrackMon.TrackPzMax            = cms.double(50.0)
TrackerCosmicTrackMon.TrackPzMin            = cms.double(-50.0)

TrackerCosmicTrackMon.doLumiAnalysis        = cms.bool(False)                       
