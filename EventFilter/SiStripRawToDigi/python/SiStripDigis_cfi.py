import FWCore.ParameterSet.Config as cms

siStripDigis = cms.EDProducer(
    "SiStripRawToDigiModule",
    ProductLabel      = cms.untracked.string('rawDataCollector'),
    CreateDigis       = cms.untracked.bool(True),
    AppendedBytes     = cms.untracked.int32(0),
    FedEventDumpFreq  = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    UseDaqRegister    = cms.untracked.bool(False),
    UseFedKey         = cms.untracked.bool(False),
    TriggerFedId      = cms.untracked.int32(0)
    )


