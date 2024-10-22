import FWCore.ParameterSet.Config as cms

SiStripDigiToRaw = cms.EDProducer(
    "SiStripDigiToRawModule",
    InputDigis=cms.InputTag('simSiStripDigis', 'ZeroSuppressed'),
    FedReadoutMode = cms.string('ZERO_SUPPRESSED'),
    PacketCode = cms.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.bool(False),
    UseWrongDigiType = cms.bool(False),
    CopyBufferHeader = cms.bool(False),
    RawDataTag = cms.InputTag('rawDataCollector')
    )
