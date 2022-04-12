import FWCore.ParameterSet.Config as cms

import sys
enableTest2 = (sys.argv[-1] != "disableTest2")
class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda accelerators: (True, -10),
                test2 = lambda accelerators: (enableTest2, -9)
            ), **kargs)

process = cms.Process("PROD1")

process.source = cms.Source("EmptySource")
if enableTest2:
    process.source.firstLuminosityBlock = cms.untracked.uint32(2)

process.maxEvents.input = 3

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSwitchProducerConditionalTaskEDAlias%d.root' % (1 if enableTest2 else 2,)),
    outputCommands = cms.untracked.vstring(
        'keep *_intProducerAlias_*_*',
        'keep *_intProducerDep_*_*',
        'keep *_intProducerDepAliasDep_*_*',
    )
)

process.intProducer1 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1))
process.intProducer2 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string("foo"),value=cms.int32(2))))
process.intProducer3 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(3))
process.intProducer4 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(4))
if enableTest2:
    process.intProducer1.throw = cms.untracked.bool(True)
else:
    process.intProducer2.throw = cms.untracked.bool(True)
    process.intProducer3.throw = cms.untracked.bool(True)
    process.intProducer4.throw = cms.untracked.bool(True)


process.intProducerAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDAlias(intProducer2 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string("")),
                                                 cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string("other"))))
)

process.intProducerDep = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer2"))
if not enableTest2:
    process.intProducerDep.labels = ["intProducer1"]

process.intProducerDepAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducerDep", "intProducerAlias")),
    test2 = cms.EDAlias(intProducer3 = cms.VPSet(cms.PSet(type = cms.string("*"), fromProductInstance = cms.string(""), toProductInstance = cms.string(""))),
                        intProducer4 = cms.VPSet(cms.PSet(type = cms.string("*"), fromProductInstance = cms.string(""), toProductInstance = cms.string("other"))))
)
process.intProducerDepAliasDep = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer3", "intProducer4"))
if not enableTest2:
    process.intProducerDepAliasDep.labels = ["intProducerAlias"]

process.ct = cms.ConditionalTask(process.intProducer1, process.intProducer2, process.intProducer3, process.intProducer4)
process.p = cms.Path(process.intProducerAlias+process.intProducerDep +
                     process.intProducerDepAlias + process.intProducerDepAliasDep,
                     process.ct)

process.ct2 = cms.ConditionalTask(process.intProducerAlias, process.intProducerDep, process.intProducerDepAlias, process.ct)
process.p2 = cms.Path(process.intProducerDepAliasDep, process.ct2)
