import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

