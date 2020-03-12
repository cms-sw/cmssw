import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#  TrackingOfflineDQM (for Tier0 Harvesting Step) ####
trackingOfflineAnalyser = DQMEDHarvester("TrackingOfflineDQM",
    GlobalStatusFilling      = cms.untracked.int32(2),
    UsedWithEDMtoMEConverter = cms.untracked.bool(True),
    TrackRatePSet            = cms.PSet(
       Name     = cms.string("NumberOfTracks_"),
       LowerCut = cms.double(0.0),
       UpperCut = cms.double(1000.0),
    ),
    TrackChi2PSet            = cms.PSet(
       Name     = cms.string("Chi2oNDF_"),
       LowerCut = cms.double(0.0),
       UpperCut = cms.double(25.0),
    ),
    TrackHitPSet            = cms.PSet(
       Name     = cms.string("NumberOfRecHitsPerTrack_"),
       LowerCut = cms.double(3.0),
       UpperCut = cms.double(30.0),
    )
)

# clone and modify modules
from DQMServices.Core.DQMQualityTester import DQMQualityTester
trackingQTesterHI = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/TrackingMonitorClient/data/tracking_qualitytest_config_tier0_heavyions.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)

# define new HI sequence
TrackingOfflineDQMClientHI = cms.Sequence(trackingQTesterHI*trackingOfflineAnalyser)

