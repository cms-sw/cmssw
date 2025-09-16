import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.output = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string('file:PoolOutputEmptyEventsMergeTest.root')
)

process.source = cms.Source("RNTupleTempSource",
                            fileNames = cms.untracked.vstring("file:PoolOutputEmptyEventsTest.root",
                                                              "file:PoolOutputTest.root"))
process.ep = cms.EndPath(process.output)
