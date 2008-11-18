import FWCore.ParameterSet.Config as cms

sipixelEDAClient = cms.EDFilter("SiPixelEDAClient",
    EventOffsetForInit = cms.untracked.int32(10),
    ActionOnLumiSection = cms.untracked.bool(False),
    ActionOnRunEnd = cms.untracked.bool(True),
    SummaryXMLFileName = cms.untracked.string('DQM/SiPixelMonitorClient/test/sipixel_offline_config.xml')
)

PixelOfflineDQMClient = cms.Sequence(sipixelEDAClient)
