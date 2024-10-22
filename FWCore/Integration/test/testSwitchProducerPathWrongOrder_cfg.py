import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda accelerators: (True, -10),
                test2 = lambda accelerators: (True, -9)
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
    fileName = cms.untracked.string('testSwitchProducerPathWrongOrder.root'),
    outputCommands = cms.untracked.vstring(
        'keep *_intProducerAlias_*_*'
    )
)

process.intProducer1 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1), throw = cms.untracked.bool(True))
process.intProducer3 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(3), values = cms.VPSet(cms.PSet(instance=cms.string("foo"),value=cms.int32(31))))

# SwitchProducer with an alias
process.intProducerAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDAlias(intProducer3 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string("")),
                                                 cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string("other"))))
)

process.t = cms.Task(process.intProducer1)
process.p = cms.Path(process.intProducerAlias + process.intProducer3, process.t)

process.e = cms.EndPath(process.out)
