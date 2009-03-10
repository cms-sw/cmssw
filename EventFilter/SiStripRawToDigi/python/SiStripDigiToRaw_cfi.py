import FWCore.ParameterSet.Config as cms

SiStripDigiToRaw = cms.EDProducer(
    "SiStripDigiToRawModule",
    InputModuleLabel = cms.string('simSiStripDigis'),
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.untracked.bool(False)
    )
