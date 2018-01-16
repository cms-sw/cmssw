import FWCore.ParameterSet.Config as cms

l1tPUM = DQMStep1Module('L1TPUM',
    regionSource = cms.InputTag("rctDigis"),
    histFolder = cms.string('L1T/L1TPUM'),
)

