import FWCore.ParameterSet.Config as cms

siStripDigis = cms.EDProducer(
    "SiStripRawToDigiModule",
    ProductLabel      = cms.InputTag('rawDataCollector'),
    AppendedBytes     = cms.int32(0),
    UseDaqRegister    = cms.bool(False),
    UseFedKey         = cms.bool(False),
    UnpackBadChannels = cms.bool(False),
    MarkModulesOnMissingFeds = cms.bool(True),
    TriggerFedId      = cms.int32(0),
    #FedEventDumpFreq  = cms.untracked.int32(0),
    #FedBufferDumpFreq = cms.untracked.int32(0),
    UnpackCommonModeValues = cms.bool(False),
    DoAllCorruptBufferChecks = cms.bool(False),
    ErrorThreshold = cms.uint32(7174)
    )


