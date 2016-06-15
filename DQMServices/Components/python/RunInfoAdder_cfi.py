import FWCore.ParameterSet.Config as cms

RunInfoAdder = cms.EDAnalyzer("RunInfoAdder",
    addRunNumber = cms.bool(True),
    addLumi = cms.bool(True),
    folder = cms.string("PixelPhase1/")
)
