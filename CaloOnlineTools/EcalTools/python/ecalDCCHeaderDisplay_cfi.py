import FWCore.ParameterSet.Config as cms

ecalDccHeaderDisplay = cms.EDAnalyzer("EcalDCCHeaderDisplay",
    EcalDCCHeaderCollection = cms.InputTag("ecalEBunpacker")
)


# foo bar baz
