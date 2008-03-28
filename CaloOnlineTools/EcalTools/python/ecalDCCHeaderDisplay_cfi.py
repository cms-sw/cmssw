import FWCore.ParameterSet.Config as cms

ecalDccHeaderDisplay = cms.EDFilter("EcalDCCHeaderDisplay",
    EcalDCCHeaderCollection = cms.InputTag("ecalEBunpacker")
)


