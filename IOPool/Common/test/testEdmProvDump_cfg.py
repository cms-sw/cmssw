import FWCore.ParameterSet.Config as cms
import argparse

parser = argparse.ArgumentParser(description='Test edmProvDump')
parser.add_argument("--lumi", type=int, default=1, help="LuminosityBlock number (default 1)")
parser.add_argument("--ivalue", type=int, default=11, help="Value for one tracked parameter (default 11)")
parser.add_argument("--output", default="testEdmProvDump.root", help="Output file name")
parser.add_argument("--accelerators", type=str, nargs='+', help="Propagated to process.options.accelerators (default is unset)")
parser.add_argument("--version", type=str, help="CMSSW version to be used in the ProcessHistory (default is unset")
args = parser.parse_args()

class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["test-one", "test-two"]
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._labels

process = cms.Process("PROD1")
if args.version:
    process._specialOverrideReleaseVersionOnlyForTesting(args.version)
if args.accelerators:
    process.add_(ProcessAcceleratorTest())
    process.options.accelerators = args.accelerators

process.source = cms.Source("IntSource",
    firstLuminosityBlock = cms.untracked.uint32(args.lumi)
)
process.maxEvents.input = 3

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(args.output),
    outputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_intProducerA_*_*'
    )
)

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")
process.DoodadESSource = cms.ESSource("DoodadESSource")

process.topLevelPSet = cms.PSet(
    someInformation = cms.string("foobar")
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("source") ),
  expectedSum = cms.untracked.int32(12),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("source", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducerU", processName=cms.InputTag.skipCurrentProcess())
  )
)

process.a2 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerA") ),
  expectedSum = cms.untracked.int32(300)
)

process.a3 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("aliasForInt") ),
  expectedSum = cms.untracked.int32(300)
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.intProducerU = cms.EDProducer("IntProducer", ivalue = cms.int32(10))

process.intProducerA = cms.EDProducer("IntProducer", ivalue = cms.int32(100))

process.aliasForInt = cms.EDAlias(
  intProducerA  = cms.VPSet(
    cms.PSet(type = cms.string('edmtestIntProduct')
    )
  )
)

process.intVectorProducer = cms.EDProducer("IntVectorProducer",
  count = cms.int32(9),
  ivalue = cms.int32(args.ivalue)
)

process.t = cms.Task(process.intProducerU, process.intProducerA, process.intVectorProducer)

process.p = cms.Path(process.intProducer * process.a1 * process.a2 * process.a3, process.t)

process.e = cms.EndPath(process.out)
