import FWCore.ParameterSet.Config as cms

#  TrackingOfflineDQM (for Tier0 Harvesting Step) ####
trackingOfflineAnalyser = cms.EDAnalyzer("TrackingOfflineDQM",
    GlobalStatusFilling      = cms.untracked.int32(2),
    UsedWithEDMtoMEConverter = cms.untracked.bool(True),
    TrackRatePSet            = cms.PSet(
        Name     = cms.string("NumberOfTracks_"),
        LowerCut = cms.double(0.0),
        UpperCut = cms.double(100.0),
    ),
    TrackChi2PSet            = cms.PSet(
        Name     = cms.string("Chi2oNDF_"),
        LowerCut = cms.double(0.0),
        UpperCut = cms.double(25.0),
    ),
    TrackHitPSet            = cms.PSet(
        Name     = cms.string("NumberOfRecHitsPerTrack_"),
        LowerCut = cms.double(3.0),
        UpperCut = cms.double(35.0),
    ),
    GoodTrackFractionPSet   = cms.PSet(
        Name     = cms.string("FractionOfGoodTracks_"),
        LowerCut = cms.double(-1.),
        UpperCut = cms.double(1.),
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

