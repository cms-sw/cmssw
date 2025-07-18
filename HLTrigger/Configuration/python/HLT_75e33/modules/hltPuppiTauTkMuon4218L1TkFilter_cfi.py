import FWCore.ParameterSet.Config as cms

hltPuppiTauTkMuon4218L1TkFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pPuppiTauTkMuon42_18')
)
