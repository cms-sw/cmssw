import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ProcessAccelerator.')

parser.add_argument("--enableTest2", help="Enable test2 accelerator", action="store_true")
parser.add_argument("--accelerators", type=str, help="Comma-separated string for accelerators to enable")

args = parser.parse_args()

class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["test1", "test2"]
        self._enabled = ["test1"]
        if args.enableTest2:
            self._enabled.append("test2")
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._enabled

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                cpu = cms.SwitchProducer.getCpu(),
                test1 = lambda accelerators: ("test1" in accelerators, 2),
                test2 = lambda accelerators: ("test2" in accelerators, 3),
            ), **kargs)

process = cms.Process("PROD1")

process.add_(ProcessAcceleratorTest())

process.source = cms.Source("EmptySource")
process.maxEvents.input = 3
if args.accelerators is not None:
    process.options.accelerators = args.accelerators.split(",")

process.intProducer1 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1))
process.intProducer2 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(2))
process.failIntProducer = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(-1), throw = cms.untracked.bool(True))

if args.enableTest2 and ("test2" in process.options.accelerators or "*" in process.options.accelerators):
    process.intProducer1.throw = cms.untracked.bool(True)
else:
    process.intProducer2.throw = cms.untracked.bool(True)

process.intProducer = SwitchProducerTest(
    cpu = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("failIntProducer")),
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer2"))
)

process.intConsumer = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer"))

process.t = cms.Task(
    process.failIntProducer,
    process.intProducer1,
    process.intProducer2,
    process.intProducer,
)
process.p = cms.Path(
    process.intConsumer,
    process.t
)
