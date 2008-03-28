import FWCore.ParameterSet.Config as cms

rctDigiToRctText = cms.EDFilter("RctDigiToRctText",
    RctInputLabel = cms.InputTag("RCTRegionSumsEmCands"),
    TextFileName = cms.string('testRctEl_'),
    HexUpperCase = cms.bool(False)
)


