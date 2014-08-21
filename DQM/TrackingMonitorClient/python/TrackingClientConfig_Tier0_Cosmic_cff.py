import FWCore.ParameterSet.Config as cms

#  TrackingOfflineDQM (for Tier0 Harvesting Step) ####
trackingOfflineAnalyser = cms.EDAnalyzer("TrackingOfflineDQM",
    GlobalStatusFilling        = cms.untracked.int32(2),
    UsedWithEDMtoMEConverter   = cms.untracked.bool(True),
    TopFolderName              = cms.untracked.string("Tracking"),                                     
    TrackingGlobalQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             dir        = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties"),
             name       = cms.string("NumberOfGoodTracks_"),
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             dir        = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties"),
             name       = cms.string("GoodTrackChi2oNDF_"),
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             dir        = cms.string("TrackParameters/highPurityTracks/pt_1/HitProperties"),
             name       = cms.string("GoodTrackNumberOfRecHitsPerTrack_"),
         ),
    ),
    TrackingLSQualityPSets = cms.VPSet(
         cms.PSet(
             QT         = cms.string("Rate"),
             LSdir      = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties/LSanalysis"),
             LSname     = cms.string("NumberOfGoodTracks_lumiFlag_"),
             LSlowerCut = cms.double( -1.0 ),
             LSupperCut = cms.double(  1.0 )    
         ),
         cms.PSet(
             QT         = cms.string("Chi2"),
             LSdir      = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties/LSanalysis"),
             LSname     = cms.string("GoodTrackChi2oNDF_lumiFlag_"),
             LSlowerCut = cms.double(  0.0 ),
             LSupperCut = cms.double( 25.0 )
         ),
         cms.PSet(
             QT         = cms.string("RecHits"),
             LSdir      = cms.string("TrackParameters/highPurityTracks/pt_1/GeneralProperties/LSanalysis"),
             LSname     = cms.string("GoodTrackNumberOfRecHitsPerTrack_lumiFlag_"),
             LSlowerCut = cms.double(  3.0 ),
             LSupperCut = cms.double( 35.0 )
         ),
    )
)

trackingQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/TrackingMonitorClient/data/tracking_qualitytest_config_tier0_cosmic.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'Tracking/TrackParameters/TrackEfficiency'
TrackEffClient.AlgoName   = 'CKFTk'

# Sequence
TrackingCosmicDQMClient = cms.Sequence(trackingQTester*trackingOfflineAnalyser*TrackEffClient)

