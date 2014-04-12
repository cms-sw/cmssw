import FWCore.ParameterSet.Config as cms

rctDigiToRctText = cms.EDAnalyzer("RctDigiToRctText",
    RctInputLabel = cms.InputTag("RCTRegionSumsEmCands"),
    TextFileName = cms.string('testRctEl_'),
    HexUpperCase = cms.bool(False)
)


