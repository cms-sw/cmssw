import FWCore.ParameterSet.Config as cms

DigiValidator = cms.EDAnalyzer(
    "SiStripDigiValidator",
    TagCollection1 = cms.untracked.InputTag("DigiSource"),
    TagCollection2 = cms.untracked.InputTag("siStripDigis","ZeroSuppressed"),
    RawCollection1 = cms.untracked.bool(False),
    RawCollection2 = cms.untracked.bool(False),
    )

# foo bar baz
# Dw90Z3GYj75Oj
# 0UJ13axYsp8oN
