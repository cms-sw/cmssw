import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("ThrowingSource", whenToThrow = cms.untracked.int32(3))
