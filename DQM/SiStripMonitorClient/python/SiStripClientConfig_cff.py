import FWCore.ParameterSet.Config as cms

#  SiStripMonitorAnalyser ####
onlineAnalyser = cms.EDFilter("SiStripAnalyser",
    StaticUpdateFrequency = cms.untracked.int32(1),
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/SiStripMonitorClient/scripts/TkMap/')
    ),
    TkMapCreationFrequency = cms.untracked.int32(1),
    SummaryCreationFrequency = cms.untracked.int32(1)
)

offlineAnalyser = cms.EDFilter("SiStripAnalyser",
    StaticUpdateFrequency = cms.untracked.int32(-1),
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/SiStripMonitorClient/scripts/TkMap/')
    ),
    TkMapCreationFrequency = cms.untracked.int32(-1),
    SummaryCreationFrequency = cms.untracked.int32(1)
)

SiStripOnlineDQMClient = cms.Sequence(onlineAnalyser)
SiStripOfflineDQMClient = cms.Sequence(offlineAnalyser)

