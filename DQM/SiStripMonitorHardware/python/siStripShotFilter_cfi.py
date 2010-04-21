import FWCore.ParameterSet.Config as cms

siStripShotFilter = cms.EDFilter(
    "SiStripShotFilter",
    OutputFilePath = cms.untracked.string('./shotChannels.dat'),
    DigiCollection = cms.InputTag("siStripDigis","ZeroSuppressed"),
    ZeroSuppressed =  cms.untracked.bool(True)
    )
