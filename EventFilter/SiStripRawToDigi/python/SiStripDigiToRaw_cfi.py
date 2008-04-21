import FWCore.ParameterSet.Config as cms

SiStripDigiToRaw = cms.EDFilter("SiStripDigiToRawModule",
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.untracked.bool(False),
    InputModuleLabel = cms.string('siStripDigis')
)


