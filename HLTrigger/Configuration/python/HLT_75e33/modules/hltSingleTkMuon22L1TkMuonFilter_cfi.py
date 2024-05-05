import FWCore.ParameterSet.Config as cms

hltSingleTkMuon22L1TkMuonFilter = cms.EDFilter("PathStatusFilter",
    logicalExpression = cms.string('pSingleTkMuon22')
)
