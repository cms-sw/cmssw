import FWCore.ParameterSet.Config as cms

rctDigiToSourceCardText = cms.EDAnalyzer("RctDigiToSourceCardText",
    RctInputLabel = cms.InputTag("L1RCTRegionSumsEmCands"),
    TextFileName = cms.string('RctDigiToSourceCardText.dat')
)


# foo bar baz
# 2zeT7BVeoqNvj
