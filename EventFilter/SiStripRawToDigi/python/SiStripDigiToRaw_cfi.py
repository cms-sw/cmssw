import FWCore.ParameterSet.Config as cms

SiStripDigiToRaw = cms.EDProducer(
    #"sistrip::DigiToRawModule",
    "SiStripDigiToRawModule",
    InputModuleLabel = cms.string('simSiStripDigis'),
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.bool(False),
    UseWrongDigiType = cms.bool(False)
    )

#SiStripDigiToRaw = cms.EDProducer(
#    "OldSiStripDigiToRawModule",
#    InputModuleLabel = cms.string('simSiStripDigis'),
#    InputDigiLabel = cms.string('ZeroSuppressed'),
#    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
#    UseFedKey = cms.untracked.bool(False)
#    )

