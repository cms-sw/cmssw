import FWCore.ParameterSet.Config as cms

FedChannelDigis = cms.EDProducer("SiStripRawToDigiModule",
    ProductLabel      = cms.untracked.string('source'),
    ProductInstance   = cms.untracked.string(''),
    CreateDigis       = cms.untracked.bool(True),
    AppendedBytes     = cms.untracked.int32(0),
    FedEventDumpFreq  = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    UseDaqRegister    = cms.untracked.bool(True),
    UseFedKey         = cms.untracked.bool(True),
    TriggerFedId      = cms.untracked.int32(-1),
    Quiet             = cms.untracked.bool(False)
)

