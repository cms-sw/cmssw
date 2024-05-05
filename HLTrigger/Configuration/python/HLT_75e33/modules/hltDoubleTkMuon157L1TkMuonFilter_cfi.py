import FWCore.ParameterSet.Config as cms

hltDoubleTkMuon157L1TkMuonFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pDoubleTkMuon15_7')
)
