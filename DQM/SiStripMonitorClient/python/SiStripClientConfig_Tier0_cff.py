import FWCore.ParameterSet.Config as cms

#  SiStripMonitorAnalyser (for Tier0 Harvesting Step) ####
siStripOfflineAnalyser = cms.EDFilter("SiStripAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(-1),
    GlobalStatusFilling      = cms.untracked.int32(2),
    TkMapCreationFrequency   = cms.untracked.int32(-1),
    SummaryCreationFrequency = cms.untracked.int32(-1),
    RawDataTag               = cms.untracked.InputTag("source"),                               
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/SiStripMonitorClient/scripts/TkMap/')
    )
)
siStripQTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    getQualityTestsFromFile = cms.untracked.bool(True)
)

# Sequence
SiStripOfflineDQMClient = cms.Sequence(siStripQTester*siStripOfflineAnalyser)

