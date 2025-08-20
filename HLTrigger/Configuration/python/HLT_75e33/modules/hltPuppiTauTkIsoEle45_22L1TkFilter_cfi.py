import FWCore.ParameterSet.Config as cms

hltPuppiTauTkIsoEle45_22L1TkFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pPuppiTauTkIsoEle45_22')
)
