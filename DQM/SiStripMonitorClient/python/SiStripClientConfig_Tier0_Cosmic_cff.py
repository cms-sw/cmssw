import FWCore.ParameterSet.Config as cms

#  SiStripOfflineDQM (for Tier0 Harvesting Step) ####
siStripOfflineAnalyser = cms.EDAnalyzer("SiStripOfflineDQM",
    GlobalStatusFilling      = cms.untracked.int32(2),
    CreateSummary            = cms.untracked.bool(False),
    SummaryConfigPath        = cms.untracked.string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml"),
    UsedWithEDMtoMEConverter = cms.untracked.bool(True),
    PrintFaultyModuleList    = cms.untracked.bool(True),
    CreateTkMap              = cms.untracked.bool(False), 
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

siStripQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0_cosmic.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'Tracking/TrackParameters/TrackEfficiency'
TrackEffClient.AlgoName   = 'CKFTk'

# Sequence
SiStripCosmicDQMClient = cms.Sequence(siStripQTester*siStripOfflineAnalyser*TrackEffClient)
#removed modules using TkDetMap
#SiStripCosmicDQMClient = cms.Sequence(siStripQTester*TrackEffClient)


# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")
SiStripDetInfoFileReade = cms.Service("SiStripDetInfoFileReader")
