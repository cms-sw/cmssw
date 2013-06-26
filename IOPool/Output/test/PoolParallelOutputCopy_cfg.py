import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTOUTPUTCOPY")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:PoolOutputTest.root')
)


process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputCopy.root')
)

process.ep = cms.EndPath(process.output)


process.options = cms.untracked.PSet(multiProcesses=cms.untracked.PSet(
        maxChildProcesses=cms.untracked.int32(3),
        maxSequentialEventsPerChild=cms.untracked.uint32(2)))

