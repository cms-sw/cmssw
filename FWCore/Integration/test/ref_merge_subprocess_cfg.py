import FWCore.ParameterSet.Config as cms
process = cms.Process("READ")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:ref_merge_prod1.root",
                                                              "file:ref_merge_prod2.root")
                            )

testProcess = cms.Process("TEST")
process.subProcess = cms.SubProcess(testProcess)

testProcess.tester = cms.EDAnalyzer("OtherThingAnalyzer",
                                other = cms.untracked.InputTag("d","testUserTag"))

testProcess.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('refInSubProcess.root')
)

testProcess.e = cms.EndPath(testProcess.tester*testProcess.out)
