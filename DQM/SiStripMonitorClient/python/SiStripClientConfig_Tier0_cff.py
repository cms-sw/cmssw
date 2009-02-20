import FWCore.ParameterSet.Config as cms

#  SiStripOfflineDQM (for Tier0 Harvesting Step) ####
siStripOfflineAnalyser = cms.EDFilter("SiStripOfflineDQM",
    GlobalStatusFilling      = cms.untracked.int32(2),
    SummaryCreationFrequency = cms.untracked.int32(-1),
    UsedWithEDMtoMEConverter = cms.untracked.bool(True)
)

siStripQTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import *
TrackEffClient.FolderName = 'SiStrip/Tracks/Efficiencies'
TrackEffClient.AlgoName   = 'CKFTk'

# Sequence
SiStripOfflineDQMClient = cms.Sequence(siStripQTester*siStripOfflineAnalyser*TrackEffClient)

