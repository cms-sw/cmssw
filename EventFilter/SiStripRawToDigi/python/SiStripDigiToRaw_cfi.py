import FWCore.ParameterSet.Config as cms

SiStripDigiToRaw = cms.EDFilter("SiStripDigiToRawModule",
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    InputModuleLabel = cms.string('siStripDigis')
)


