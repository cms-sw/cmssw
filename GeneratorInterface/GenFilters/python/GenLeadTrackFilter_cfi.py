import FWCore.ParameterSet.Config as cms

genLeadTrackFilter = cms.EDFilter('GenLeadTrackFilter',
  HepMCProduct             = cms.InputTag("generator"),
  GenLeadTrackPt           = cms.double(12),
  GenEta                   = cms.double(2.5)
)
