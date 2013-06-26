import FWCore.ParameterSet.Config as cms

DigiValidator = cms.EDAnalyzer(
    "SiStripDigiValidator",
    TagCollection1 = cms.untracked.InputTag("DigiSource"),
    TagCollection2 = cms.untracked.InputTag("siStripDigis","ZeroSuppressed"),
    RawCollection1 = cms.untracked.bool(False),
    RawCollection2 = cms.untracked.bool(False),
    )

