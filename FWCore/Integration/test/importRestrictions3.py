# This succeeds because allowed is allowed to import "restricted"

import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.load("FWCore.Integration.allowed_cff")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
