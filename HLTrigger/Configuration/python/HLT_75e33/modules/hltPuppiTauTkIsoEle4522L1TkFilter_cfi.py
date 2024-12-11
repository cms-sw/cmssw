import FWCore.ParameterSet.Config as cms

hltPuppiTauTkIsoEle4522L1TkFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pPuppiTauTkIsoEle45_22')
)
