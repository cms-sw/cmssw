import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ProcessAccelerator.')

parser.add_argument("--enableOther", action="store_true", help="Enable other accelerator")
parser.add_argument("--setInResolver", type=str, default="", help="Set the default variant in module type resolver")
parser.add_argument("--setInModule", type=str, default="", help="Set the default variant in module itself")
parser.add_argument("--accelerators", type=str, help="Comma-separated string for accelerators to enable")
parser.add_argument("--expectOther", action="store_true", help="Set this if the 'other' variant is expected to get run")

args = parser.parse_args()

class ModuleTypeResolverTest:
    def __init__(self, accelerators):
        self._variants = []
        if "other" in accelerators:
            self._variants.append("other")
        if "cpu" in accelerators:
            self._variants.append("cpu")
        if args.setInResolver != "":
            if args.setInResolver not in self._variants:
                raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "Trying to set variant globally in ModuleTypeResolverTest to {}, but a corresponding accelerator is not available".format(args.setInResolver))
            self._variants.remove(args.setInResolver)
            self._variants.insert(0, args.setInResolver)
        if len(self._variants) == 0:
            raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "No 'cpu' or 'other' accelerator available")

    def plugin(self):
        return "edm::test::ConfigurableTestTypeResolverMaker"

    def setModuleVariant(self, module):
        if module.type_().startswith("generic::"):
            if hasattr(module, "variant"):
                if module.variant.value() not in self._variants:
                    raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "Module {} has the Test variant set explicitly to {}, but its accelerator is not available for the job".format(module.label_(), module.variant.value()))
            else:
                module.variant = cms.untracked.string(self._variants[0])

class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["other"]
        self._enabled = []
        if args.enableOther:
            self._enabled.append("other")
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._enabled
    def moduleTypeResolver(self, accelerators):
        return ModuleTypeResolverTest(accelerators)

process = cms.Process("PROD1")

process.add_(ProcessAcceleratorTest())

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(1)
)
process.maxEvents.input = 3
if args.accelerators is not None:
    process.options.accelerators = args.accelerators.split(",")

# EventSetup
process.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3),
    iovIsRunNotTime = cms.bool(True)
)

process.esTestProducerA = cms.ESProducer("generic::ESTestProducerA", valueCpu = cms.int32(10), valueOther = cms.int32(20))

process.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,3),
    expectedValues=cms.untracked.vint32(11,12,13)
)

# Event
process.intProducer = cms.EDProducer("generic::IntProducer", valueCpu = cms.int32(1), valueOther = cms.int32(2))

process.intConsumer = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustMatch = cms.untracked.int32(1)
)

if args.setInModule != "":
    process.esTestProducerA.variant = cms.untracked.string(args.setInModule)
    process.intProducer.variant = cms.untracked.string(args.setInModule)

if args.expectOther:
    process.esTestAnalyzerA.expectedValues = [21, 22, 23]
    process.intConsumer.valueMustMatch = 2

process.t = cms.Task(
    process.intProducer
)
process.p = cms.Path(
    process.intConsumer + process.esTestAnalyzerA,
    process.t
)
