import FWCore.ParameterSet.Config as cms

# WARNING: PPRESENTLY USING OLD DIGI-TO-RAW MODULE!!!
# WARNING: SHOULD MIGRATE TO NEW sistrip::DigiToRaw MODULE ONCE VALIDATED!

""" #Config for sistrip::DigiToRaw
SiStripDigiToRaw = cms.EDProducer(
    "sistrip::DigiToRawModule",
    InputModuleLabel = cms.string('simSiStripDigis'),
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.bool(False),
    UseWrongDigiType = cms.bool(False)
    )
"""

SiStripDigiToRaw = cms.EDProducer(
    "OldSiStripDigiToRawModule",
    InputModuleLabel = cms.string('simSiStripDigis'),
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.untracked.bool(False)
    )

