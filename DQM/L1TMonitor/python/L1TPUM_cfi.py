import FWCore.ParameterSet.Config as cms

l1tPUM = cms.EDAnalyzer("L1TPUM",
    regionSource = cms.InputTag("rctHwDigis"),
    histFolder = cms.string('L1T/L1TPUM'),
)

