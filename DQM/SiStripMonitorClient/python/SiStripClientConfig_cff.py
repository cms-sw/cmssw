import FWCore.ParameterSet.Config as cms

#  SiStripMonitorAnalyser ####
# for Online running
onlineAnalyser = cms.EDFilter("SiStripAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(1),
    GlobalStatusFilling      = cms.untracked.bool(True),
    TkMapCreationFrequency   = cms.untracked.int32(1),
    SummaryCreationFrequency = cms.untracked.int32(1),
    RawDataTag               = cms.untracked.InputTag("source"),                              
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/SiStripMonitorClient/scripts/TkMap/')
    )
)

# for Offline running
offlineAnalyser = cms.EDFilter("SiStripAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(-1),
    GlobalStatusFilling      = cms.untracked.bool(True),
    TkMapCreationFrequency   = cms.untracked.int32(-1),
    SummaryCreationFrequency = cms.untracked.int32(1),
    RawDataTag               = cms.untracked.InputTag("source"),                               
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/SiStripMonitorClient/scripts/TkMap/')
    )
)

# Sequence
SiStripOnlineDQMClient = cms.Sequence(onlineAnalyser)
SiStripOfflineDQMClient = cms.Sequence(offlineAnalyser)

