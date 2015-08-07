import FWCore.ParameterSet.Config as cms

l1tPUM = cms.EDAnalyzer("L1TPUM",
    regionSource = cms.InputTag("rctDigis"),
    histFolder = cms.string('L1T/L1TPUM'),
)

