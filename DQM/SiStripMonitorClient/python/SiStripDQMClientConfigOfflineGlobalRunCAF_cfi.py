import FWCore.ParameterSet.Config as cms

# SiStrip DQM Client
SiStripDQMClientGlobalRunCAF = cms.EDFilter("SiStripAnalyser",
    StaticUpdateFrequency = cms.untracked.int32(-1),
    GlobalStatusFilling = cms.untracked.bool(True),
    TkMapCreationFrequency = cms.untracked.int32(-1),
    SummaryCreationFrequency = cms.untracked.int32(-1),
    TkmapParameters = cms.PSet(
        loadFedCabling = cms.untracked.bool(True),
        trackerdatPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/Integration/test/TkMap/')
    )
)


