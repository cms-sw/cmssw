import FWCore.ParameterSet.Config as cms

lhemttFilter = cms.EDFilter("LHEmttFilter",
    src = cms.InputTag("source"),
    MinInvMass = cms.double(1000),
    ptMin = cms.double(0)
)
