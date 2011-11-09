import FWCore.ParameterSet.Config as cms

modelSelector = cms.EDFilter('ModelFilter',
                             source = cms.InputTag("source"),
                             modelTag = cms.string("T2tt"),
                             parameterMins = cms.vdouble(-99999999,-99999999),
                             parameterMaxs = cms.vdouble(99999999, 99999999)
)
