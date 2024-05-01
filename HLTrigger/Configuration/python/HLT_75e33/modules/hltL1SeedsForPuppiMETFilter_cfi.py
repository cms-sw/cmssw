import FWCore.ParameterSet.Config as cms

hltL1SeedsForPuppiMETFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pPuppiMET200')
)
