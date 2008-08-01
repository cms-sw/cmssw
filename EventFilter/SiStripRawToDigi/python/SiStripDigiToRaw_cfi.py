import FWCore.ParameterSet.Config as cms

SiStripDigiToRaw = cms.EDProducer("SiStripDigiToRawModule",
                                    InputDigiLabel = cms.string('ZeroSuppressed'),
                                    FedReadoutMode = cms.untracked.string('ZERO_SUPPRESSED'),
                                    UseFedKey = cms.untracked.bool(False),
                                    InputModuleLabel = cms.string('simSiStripDigis')
                                )
