import FWCore.ParameterSet.Config as cms

PoolOutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PoolOutput.root')
)


