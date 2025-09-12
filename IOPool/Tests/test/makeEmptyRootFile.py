import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 10

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring("drop *"),
                               fileName = cms.untracked.string("empty.root"))

process.o = cms.EndPath(process.out)
