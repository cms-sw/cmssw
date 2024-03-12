import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("ThrowingSource", whenToThrow = cms.untracked.int32(3))
# foo bar baz
# TQXFWeUVHCGwV
# DXekD6O6F67FH
