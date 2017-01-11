import FWCore.ParameterSet.Config as cms

RunInfoAdder = cms.EDAnalyzer("RunInfoAdder",
    addRunNumber = cms.bool(True),
    addLumi = cms.bool(False),
    # apply only to these folders. Add "" for all.
    folder = cms.vstring("Pixel", "SiStrip", "Tracking")
)
