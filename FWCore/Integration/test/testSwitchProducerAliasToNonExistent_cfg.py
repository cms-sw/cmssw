import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = cms.Process("PROD1")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSwitchProducerAliasOutput.root'),
    outputCommands = cms.untracked.vstring(
        'keep *_intProducer3_*_*',
        'keep *_intProducerAlias__*'
    )
)

process.intProducer1 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1), throw = cms.untracked.bool(True))
process.intProducer4 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(42), throw = cms.untracked.bool(True))

process.intProducerAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDAlias(intProducer4 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string(""))),
                        intProducer3 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string(""))
))
)

process.t = cms.Task(process.intProducerAlias, process.intProducer1, process.intProducer4)
process.p = cms.Path(process.t)

process.e = cms.EndPath(process.out)
