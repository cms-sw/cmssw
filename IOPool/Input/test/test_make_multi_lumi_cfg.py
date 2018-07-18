import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITE")

process.source = cms.Source("EmptySource", numberEventsInLuminosityBlock = cms.untracked.uint32(4))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20))

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("multi_lumi.root"))

process.o = cms.EndPath(process.out)
