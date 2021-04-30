import FWCore.ParameterSet.Config as cms

lhemttFilter = cms.EDFilter("LHEmttFilter",
    src = cms.InputTag("source"),
    MinInvMass = cms.double(1000),
    MaxInvMass = cms.double(-1),
    ptMin = cms.double(0)
)
