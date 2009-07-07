# This succeeds because it is importing "restricted" at the top level

import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.load("FWCore.Integration.restricted_cff")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
