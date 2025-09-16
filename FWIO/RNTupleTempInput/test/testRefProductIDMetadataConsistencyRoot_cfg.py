import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ModuleTypeResolver and Ref')
parser.add_argument("--enableOther", action="store_true", help="Enable other accelerator. Also sets Lumi to 2")
args = parser.parse_args()

class ModuleTypeResolverTest:
    def __init__(self, accelerators):
        self._variants = []
        if "other" in accelerators:
            self._variants.append("other")
        if "cpu" in accelerators:
            self._variants.append("cpu")
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
    firstEvent = cms.untracked.uint32(10 if args.enableOther else 1),
)
process.maxEvents.input = 3

process.intprod = cms.EDProducer("generic::IntTransformer", valueCpu = cms.int32(1), valueOther = cms.int32(2))

process.thing = cms.EDProducer("ThingProducer")

process.otherThing = cms.EDProducer("OtherThingProducer",
    thingTag=cms.InputTag("thing")
)

process.t = cms.Task(
    process.intprod,
    process.thing,
    process.otherThing
)
process.p = cms.Path(process.t)

fname = "refconsistency_{}".format(process.source.firstEvent.value())
process.out = cms.OutputModule("RNTupleTempOutputModule",
    fileName = cms.untracked.string(fname+".root")
)

process.ep = cms.EndPath(process.out)
