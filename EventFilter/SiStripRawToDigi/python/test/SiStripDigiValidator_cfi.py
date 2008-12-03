import FWCore.ParameterSet.Config as cms

DigiValidator = cms.EDFilter("SiStripDigiValidator",
    FedRawDataMode = cms.untracked.bool(False),
    Collection2 = cms.untracked.InputTag("SiStripDigis","ZeroSuppressed"),
    Collection1 = cms.untracked.InputTag("SiStripDigiSource")
)


