import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tPUM = DQMEDAnalyzer('L1TPUM',
    regionSource = cms.InputTag("rctDigis"),
    histFolder = cms.string('L1T/L1TPUM'),
)

