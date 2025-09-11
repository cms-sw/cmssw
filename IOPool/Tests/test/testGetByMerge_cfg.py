import FWCore.ParameterSet.Config as cms

process = cms.Process("MERGE")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testGetBy1.root',
        'file:testGetBy1Mod.root'
    )
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducer"),
                                       cms.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess())
  ),
  expectedSum = cms.untracked.int32(27),
  runProducerParameterCheck = cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGetByMerge.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(3))

process.t = cms.Task(process.intProducer)

process.p = cms.Path(process.a1, process.t)

process.e = cms.EndPath(process.out)
