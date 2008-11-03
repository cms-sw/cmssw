import FWCore.ParameterSet.Config as cms

dqmFEDIntegrity = cms.EDFilter("DQMFEDIntegrityClient")

DQMFEDIntegrityClient = cms.Sequence(dqmFEDIntegrity)
