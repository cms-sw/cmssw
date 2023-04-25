import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test SwitchProducer in Task.')
parser.add_argument("--disableTest2", help="Disable test2 SwitchProducer case", action="store_true")
parser.add_argument("--input", help="Read input file from a previous step of this same configuration", action="store_true")
parser.add_argument("--conditionalTask", help="Use ConditionalTask instead of Task", action="store_true")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

enableTest2 = not args.disableTest2
class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda accelerators: (True, -10),
                test2 = lambda accelerators: (enableTest2, -9)
            ), **kargs)

process = cms.Process("PROD2" if args.input else "PROD1")

if args.input:
     # Test that having SwitchProducers with same labels as products from
    # earlier processes works
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring("file:testSwitchProducer{}Task{}.root".format(
            "Conditional" if args.conditionalTask else "",
            1 if enableTest2 else 2,
        ))
    )
    process.maxEvents.input = -1
else:
    process.source = cms.Source("EmptySource")
    process.maxEvents.input = 3
    if enableTest2:
        process.source.firstLuminosityBlock = cms.untracked.uint32(2)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSwitchProducer{}Task{}{}.root'.format(
        "Conditional" if args.conditionalTask else "",
        "Input" if args.input else "",
        1 if enableTest2 else 2,
    )),
    outputCommands = cms.untracked.vstring(
        'keep *_intProducer_*_*',
        'keep *_intProducerOther_*_*',
        'keep *_intProducerAlias_*_*',
        'keep *_intProducerAlias2_other_*',
        'keep *_intProducerDep1_*_*',
        'keep *_intProducerDep2_*_*',
        'keep *_intProducerDep3_*_*',
    )
)

process.intProducer1 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1))
process.intProducer2 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(2))
process.intProducer3 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string("foo"),value=cms.int32(2))))
process.intProducer4 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(42), throw = cms.untracked.bool(True))
process.intProducer5 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(3), values = cms.VPSet(cms.PSet(instance=cms.string("foo"),value=cms.int32(3))))
if enableTest2:
    process.intProducer1.throw = cms.untracked.bool(True)
else:
    process.intProducer2.throw = cms.untracked.bool(True)
    process.intProducer3.throw = cms.untracked.bool(True)
    process.intProducer5.throw = cms.untracked.bool(True)

process.intProducer = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer2"))
)
# Test also existence of another SwitchProducer here
process.intProducerOther = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer2"))
)
# SwitchProducer with an alias
process.intProducerAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDAlias(intProducer3 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string("")),
                                                 cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string("other"))))
)

process.intProducerAlias2 = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDAlias(intProducer4 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string(""))),
                        intProducer5 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string("other"))))
)

# Test multiple consumers of a SwitchProducer
process.intProducerDep1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer"))
process.intProducerDep2 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer"))
process.intProducerDep3 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer"))

if args.conditionalTask:
    process.ct = cms.ConditionalTask(process.intProducer, process.intProducerOther, process.intProducerAlias, process.intProducerAlias2,
                                     process.intProducer1, process.intProducer2, process.intProducer3, process.intProducer4, process.intProducer5)
    process.p = cms.Path(process.intProducerDep1+process.intProducerDep2+process.intProducerDep3, process.ct)
else:
    process.t = cms.Task(process.intProducer, process.intProducerOther, process.intProducerAlias, process.intProducerAlias2,
                         process.intProducerDep1, process.intProducerDep2, process.intProducerDep3,
                         process.intProducer1, process.intProducer2, process.intProducer3, process.intProducer4, process.intProducer5)
    process.p = cms.Path(process.t)

process.e = cms.EndPath(process.out)
