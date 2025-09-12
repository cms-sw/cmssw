import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.intProducerA = cms.EDProducer("IntProducer", ivalue = cms.int32(100))

process.aliasForInt = cms.EDAlias(
    intProducerA  = cms.VPSet(
        cms.PSet(type = cms.string('edmtestIntProduct')
        )
    )
)

process.testout = cms.OutputModule("TestGlobalOutput",
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_intProducerA_*_*"
    )
)

process.testoutlimited = cms.OutputModule("TestLimitedOutput",
    outputCommands = cms.untracked.vstring(
        "keep *",
        "drop *_intProducerA_*_*"
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testOutput1.root'),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_intProducerA_*_*'
    )
)

process.path = cms.Path(process.intProducerA)

process.endpath = cms.EndPath(process.testout + process.testoutlimited + process.out)
