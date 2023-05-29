import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description="Test SwitchProducer, that has an EDAlias with '*' wildcard, in a ConditionalTask")

parser.add_argument("--disableTest2", help="Disable 'test2' case of the SwitchProducerTest", action="store_true")
parser.add_argument("--wildcardOnOtherModule", help="Use the wildcard for alias from another module", action="store_true")

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

process = cms.Process("PROD1")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.intProducer1 = cms.EDProducer(
    "ManyIntProducer",
    ivalue = cms.int32(1),
    values = cms.VPSet(
    )
)
process.intProducer2 = cms.EDProducer(
    "ManyIntProducer",
    ivalue = cms.int32(11),
    values = cms.VPSet(
        cms.PSet(instance=cms.string("bar"), value=cms.int32(12))
    )
)
process.intProducer3 = cms.EDProducer(
    "ManyIntProducer",
    ivalue = cms.int32(21)
)
if args.wildcardOnOtherModule:
    process.intProducer1.values.append(cms.PSet(instance=cms.string("bar"), value=cms.int32(2)))
    process.intProducer2.values = []

process.intProducer4 = cms.EDProducer(
    "ManyIntProducer",
    ivalue = cms.int32(31),
    values = cms.VPSet(
        cms.PSet(instance=cms.string("foo"), value=cms.int32(32)),
        cms.PSet(instance=cms.string("bar"), value=cms.int32(33)),
        cms.PSet(instance=cms.string("xyzzy"), value=cms.int32(34)),
   )
)

process.intProducer = SwitchProducerTest(
    test1 = cms.EDAlias(
        intProducer4 = cms.EDAlias.allProducts()
    ),
    test2 = cms.EDAlias()
)
allMatchName = "intProducer1"
otherName ="intProducer2"
if args.wildcardOnOtherModule:
    (allMatchName, otherName) = (otherName, allMatchName)
setattr(process.intProducer.test2, allMatchName, cms.EDAlias.allProducts())
setattr(process.intProducer.test2, otherName, cms.VPSet(
    cms.PSet(type = cms.string("*"), fromProductInstance = cms.string(""), toProductInstance = cms.string("foo")),
    cms.PSet(type = cms.string("*"), fromProductInstance = cms.string("bar"), toProductInstance = cms.string("*"))
))
process.intProducer.test2.intProducer3 = cms.VPSet(
    cms.PSet(type = cms.string("edmtestIntProduct"), toProductInstance = cms.string("xyzzy"))
)

process.intConsumer = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer", srcEvent = cms.untracked.VInputTag("intProducer"))
process.intConsumer2 = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer", srcEvent = cms.untracked.VInputTag("intProducer", "intProducer2", "intProducer3"))

process.ct = cms.ConditionalTask(process.intProducer1, process.intProducer2, process.intProducer3, process.intProducer4, process.intProducer)

process.p1 = cms.Path(process.intConsumer, process.ct)
process.p2 = cms.Path(process.intConsumer2, process.ct)
