import FWCore.ParameterSet.Config as cms

siStripDigis = cms.EDFilter("SiStripRawToDigiModule",
    ProductLabel = cms.untracked.string('rawDataCollector'),
    AppendedBytes = cms.untracked.int32(0),
    UseFedKey = cms.untracked.bool(False),
    FedEventDumpFreq = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    TriggerFedId = cms.untracked.int32(0),
    CreateDigis = cms.untracked.bool(True)
)


