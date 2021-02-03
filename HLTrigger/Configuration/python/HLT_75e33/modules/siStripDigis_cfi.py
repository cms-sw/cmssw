import FWCore.ParameterSet.Config as cms

siStripDigis = cms.EDProducer("SiStripRawToDigiModule",
    AppendedBytes = cms.int32(0),
    DoAPVEmulatorCheck = cms.bool(False),
    DoAllCorruptBufferChecks = cms.bool(False),
    ErrorThreshold = cms.uint32(7174),
    LegacyUnpacker = cms.bool(False),
    MarkModulesOnMissingFeds = cms.bool(True),
    ProductLabel = cms.InputTag("rawDataCollector"),
    TriggerFedId = cms.int32(0),
    UnpackBadChannels = cms.bool(False),
    UnpackCommonModeValues = cms.bool(False),
    UseDaqRegister = cms.bool(False),
    UseFedKey = cms.bool(False)
)
