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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSwitchProducerPath%d.root' % (1 if enableTest2 else 2,)),
    outputCommands = cms.untracked.vstring(
        'keep *_intProducer_*_*',
        'keep *_intProducerAlias_*_*',
        'keep *_intProducerDep1_*_*',
        'keep *_intProducerDep2_*_*',
        'keep *_intProducerDep3_*_*',
    )
)

process.intProducer1 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1))
process.intProducer2 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(2))
process.intProducer3 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string("foo"),value=cms.int32(2))))
if enableTest2:
    process.intProducer1.throw = cms.untracked.bool(True)
else:
    process.intProducer2.throw = cms.untracked.bool(True)
    process.intProducer3.throw = cms.untracked.bool(True)

process.intProducer = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer2"))
)
# SwitchProducer with an alias
process.intProducerAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDAlias(intProducer3 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string("")),
                                                 cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string("other"))))
)

# SwitchProducer of EDAlias in a Path causes alone is enough to trigger aliased-for EDProducers to be run
if enableTest2:
    process.intProducer4 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(314), throw=cms.untracked.bool(True))
    process.intProducer6 = cms.EDProducer("edmtest::MustRunIntProducer", ivalue = cms.int32(4))
else:
    process.intProducer4 = cms.EDProducer("edmtest::MustRunIntProducer", ivalue = cms.int32(314))
    process.intProducer6 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(4), throw=cms.untracked.bool(True))
process.intProducer5 = process.intProducer4.clone(ivalue = 159)
process.intProducer7 = process.intProducer6.clone(ivalue = 2)

process.intProducerAlias2 = SwitchProducerTest(
    test1 = cms.EDAlias(intProducer4 = cms.VPSet(cms.PSet(type = cms.string("*"), fromProductInstance = cms.string(""), toProductInstance = cms.string(""))),
                        intProducer5 = cms.VPSet(cms.PSet(type = cms.string("*"), fromProductInstance = cms.string(""), toProductInstance = cms.string("other")))),
    test2 = cms.EDAlias(intProducer6 = cms.VPSet(cms.PSet(type = cms.string("*"), fromProductInstance = cms.string(""), toProductInstance = cms.string(""))),
                        intProducer7 = cms.VPSet(cms.PSet(type = cms.string("*"), fromProductInstance = cms.string(""), toProductInstance = cms.string("other"))))
)


# Test multiple consumers of a SwitchProducer
process.intProducerDep1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer"))
process.intProducerDep2 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer"))
process.intProducerDep3 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer"))


process.t = cms.Task(process.intProducer1, process.intProducer2, process.intProducer3,
                     process.intProducerDep1, process.intProducerDep2, process.intProducerDep3)
process.p = cms.Path(process.intProducer + process.intProducerAlias, process.t)

process.t2 = cms.Task(process.intProducer4, process.intProducer5, process.intProducer6, process.intProducer7)
process.p2 = cms.Path(process.intProducerAlias2, process.t2)

process.e = cms.EndPath(process.out)
