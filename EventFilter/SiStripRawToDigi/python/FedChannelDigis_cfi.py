import FWCore.ParameterSet.Config as cms

FedChannelDigis = cms.EDFilter("SiStripRawToDigiModule",
    ProductLabel = cms.untracked.string('source'),
    AppendedBytes = cms.untracked.int32(0),
    UseFedKey = cms.untracked.bool(True),
    FedEventDumpFreq = cms.untracked.int32(0),
    Quiet = cms.untracked.bool(False),
    FedBufferDumpFreq = cms.untracked.int32(0),
    TriggerFedId = cms.untracked.int32(-1),
    ProductInstance = cms.untracked.string(''),
    CreateDigis = cms.untracked.bool(True)
)


