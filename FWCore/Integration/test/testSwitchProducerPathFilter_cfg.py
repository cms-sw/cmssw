import FWCore.ParameterSet.Config as cms

import sys
enableTest2 = (sys.argv[-1] != "disableTest2")
class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (enableTest2, -9)
            ), **kargs)

process = cms.Process("PROD1")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource")
if enableTest2:
    process.source.firstLuminosityBlock = cms.untracked.uint32(2)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSwitchProducerPathFilter%d.root' % (1 if enableTest2 else 2,)),
    outputCommands = cms.untracked.vstring(
        'keep *_intProducer_*_*'
    )
)

process.intProducer1 = cms.EDProducer("FailingProducer")
process.intProducer2 = cms.EDProducer("FailingProducer")
process.intProducer3 = cms.EDProducer("ManyIntProducer",
    ivalue = cms.int32(3),
    values = cms.VPSet(cms.PSet(instance=cms.string("foo"),value=cms.int32(31))),
    throw = cms.untracked.bool(True)
)

process.intProducer = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.vstring("intProducer1")),
    test2 = cms.EDProducer("AddIntsProducer", labels = cms.vstring("intProducer2"))
)
# SwitchProducer with an alias
process.intProducerAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.vstring("intProducer1")),
    test2 = cms.EDAlias(intProducer3 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string("")),
                                                 cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string("other"))))
)


process.f = cms.EDFilter("TestFilterModule", acceptValue = cms.untracked.int32(-1))

process.t = cms.Task(process.intProducer1, process.intProducer2, process.intProducer3)
process.p = cms.Path(process.f+process.intProducer, process.t)

process.e = cms.EndPath(process.out)
