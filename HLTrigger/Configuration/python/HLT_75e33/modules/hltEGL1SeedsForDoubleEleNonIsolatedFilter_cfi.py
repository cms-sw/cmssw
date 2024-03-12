import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForDoubleEleNonIsolatedFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pDoubleEGEle37_24 or pDoubleTkEle25_12')
)
# foo bar baz
# PB2RxLHci5l3z
# P4T3saIzdG3iR
