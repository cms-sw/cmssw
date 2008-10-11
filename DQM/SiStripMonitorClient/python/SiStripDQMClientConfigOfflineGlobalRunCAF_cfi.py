import FWCore.ParameterSet.Config as cms

# SiStrip DQM Client

SiStripDQMClientGlobalRunCAF = cms.EDAnalyzer("SiStripAnalyser",
    StaticUpdateFrequency    = cms.untracked.int32(-1),
    TkMapCreationFrequency   = cms.untracked.int32(-1),
    SummaryCreationFrequency = cms.untracked.int32(-1),
    GlobalStatusFilling      = cms.untracked.int32(1),
    RawDataTag               = cms.untracked.InputTag("source"),
    TkmapParameters          = cms.PSet(
        trackerdatPath    = cms.untracked.string('CommonTools/TrackerMap/data/'),
        trackermaptxtPath = cms.untracked.string('DQM/Integration/test/TkMap/'),
        loadFedCabling    = cms.untracked.bool(True)
    )
)
