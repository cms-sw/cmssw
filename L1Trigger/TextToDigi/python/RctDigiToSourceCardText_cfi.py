import FWCore.ParameterSet.Config as cms

rctDigiToSourceCardText = cms.EDFilter("RctDigiToSourceCardText",
    RctInputLabel = cms.InputTag("L1RCTRegionSumsEmCands"),
    TextFileName = cms.string('RctDigiToSourceCardText.dat')
)


