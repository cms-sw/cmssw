import FWCore.ParameterSet.Config as cms

# WARNING: PPRESENTLY USING OLD DIGI-TO-RAW MODULE!!!
# WARNING: SHOULD MIGRATE TO NEW sistrip::DigiToRaw MODULE ONCE VALIDATED!

SiStripDigiToRaw = cms.EDProducer(
    "OldSiStripDigiToRawModule",
    InputModuleLabel = cms.string('simSiStripDigis'),
    InputDigiLabel = cms.string('ZeroSuppressed'),
    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
    UseFedKey = cms.untracked.bool(False)
    )

