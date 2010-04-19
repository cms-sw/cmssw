import FWCore.ParameterSet.Config as cms

siStripShotFilter = cms.EDFilter(
    "SiStripShotFilter",
    OutputFilePath = cms.string('./shotChannels.dat'),
    DigiCollection = cms.InputTag("siStripDigis","ZeroSuppressed"),
    ZeroSuppressed =  cms.bool(True)
    )
