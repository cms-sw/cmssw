import FWCore.ParameterSet.Config as cms
process = cms.Process("READ")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:ref_merge_prod1.root",
                                                              "file:ref_merge_prod2.root")
                            )

testProcess = cms.Process("TEST")
process.subProcess = cms.SubProcess(testProcess)

testProcess.a = cms.EDProducer("IntProducer",
                           ivalue = cms.int32(1))

testProcess.tester = cms.EDAnalyzer("OtherThingAnalyzer",
                                other = cms.untracked.InputTag("d","testUserTag"))

testProcess.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('refInSubProcess.root')
)

testProcess.p = cms.Path(testProcess.a)

testProcess.e = cms.EndPath(testProcess.tester*testProcess.out)

testProcessA = cms.Process("TESTA")
testProcess.subProcess = cms.SubProcess(testProcessA)

testProcessA.a = cms.EDProducer("IntProducer",
                           ivalue = cms.int32(1))

testProcessA.tester = cms.EDAnalyzer("OtherThingAnalyzer",
                                other = cms.untracked.InputTag("d","testUserTag"))

testProcessA.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('refInSubProcessA.root')
)

testProcessA.p = cms.Path(testProcessA.a)

testProcessA.e = cms.EndPath(testProcessA.tester*testProcessA.out)
