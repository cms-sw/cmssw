import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test various Alpaka module types')

parser.add_argument("--accelerators", type=str, help="Set process.options.accelerators (comma-separated string, default is to use default)", default="")
parser.add_argument("--moduleBackend", type=str, help="Set Alpaka backend for module instances", default="")
parser.add_argument("--expectBackend", type=str, help="Expect this backend to run")
parser.add_argument("--run", type=int, help="Run number (default: 1)", default=1)

args = parser.parse_args()

process = cms.Process('TEST')

process.source = cms.Source('EmptySource',
    firstRun = cms.untracked.uint32(args.run)
)

process.maxEvents.input = 10

if len(args.accelerators) != 0:
    process.options.accelerators = args.accelerators.split(",")

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load("HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

process.alpakaESRecordASource = cms.ESSource("EmptyESSource",
    recordName = cms.string('AlpakaESTestRecordA'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
process.alpakaESRecordBSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('AlpakaESTestRecordB'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
process.alpakaESRecordCSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('AlpakaESTestRecordC'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.esProducerA = cms.ESProducer("cms::alpakatest::TestESProducerA", value = cms.int32(42))
process.esProducerB = cms.ESProducer("cms::alpakatest::TestESProducerB", value = cms.int32(314159))
process.esProducerC = cms.ESProducer("cms::alpakatest::TestESProducerC", value = cms.int32(27))

from HeterogeneousCore.AlpakaTest.testAlpakaESProducerA_cfi import testAlpakaESProducerA 
process.alpakaESProducerA = testAlpakaESProducerA.clone(appendToDataLabel = cms.string("appendedLabel"))
process.alpakaESProducerB = cms.ESProducer("TestAlpakaESProducerB@alpaka", explicitLabel = cms.string("explicitLabel"))
process.alpakaESProducerC = cms.ESProducer("TestAlpakaESProducerC@alpaka")
process.alpakaESProducerD = cms.ESProducer("TestAlpakaESProducerD@alpaka",
    srcA = cms.ESInputTag("", "appendedLabel"),
    srcB = cms.ESInputTag("", "explicitLabel"),
)

process.intProduct = cms.EDProducer("IntProducer", ivalue = cms.int32(42))

from HeterogeneousCore.AlpakaTest.testAlpakaGlobalProducer_cfi import testAlpakaGlobalProducer
process.alpakaGlobalProducer = testAlpakaGlobalProducer.clone(
    eventSetupSource = cms.ESInputTag("alpakaESProducerA", "appendedLabel"),
    size = dict(
        alpaka_serial_sync = 10,
        alpaka_cuda_async = 20
    )
)
process.alpakaStreamProducer = cms.EDProducer("TestAlpakaStreamProducer@alpaka",
    source = cms.InputTag("intProduct"),
    eventSetupSource = cms.ESInputTag("alpakaESProducerB", "explicitLabel"),
    size = cms.PSet(
        alpaka_serial_sync = cms.int32(5),
        alpaka_cuda_async = cms.int32(25)
    )
)
process.alpakaStreamInstanceProducer = cms.EDProducer("TestAlpakaStreamProducer@alpaka",
    source = cms.InputTag("intProduct"),
    eventSetupSource = cms.ESInputTag("alpakaESProducerB", "explicitLabel"),
    productInstanceName = cms.string("testInstance"),
    size = cms.PSet(
        alpaka_serial_sync = cms.int32(6),
        alpaka_cuda_async = cms.int32(36)
    )
)
process.alpakaStreamSynchronizingProducer = cms.EDProducer("TestAlpakaStreamSynchronizingProducer@alpaka",
    source = cms.InputTag("alpakaGlobalProducer"),
    intSource = cms.InputTag("intProduct"),
    expectedInt = cms.int32(84) # sum of intProduct and esProducerA
)

process.alpakaGlobalConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaGlobalProducer"),
    expectSize = cms.int32(10)
)
process.alpakaStreamConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamProducer"),
    expectSize = cms.int32(5)
)
process.alpakaStreamInstanceConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamInstanceProducer", "testInstance"),
    expectSize = cms.int32(6)
)
process.alpakaStreamSynchronizingConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamSynchronizingProducer"),
    expectSize = cms.int32(10)
)

if args.moduleBackend != "":
    for name in ["ESProducerA", "ESProducerB", "ESProducerC", "ESProducerD",
                 "GlobalProducer", "StreamProducer", "StreamInstanceProducer", "StreamSynchronizingProducer"]:
        mod = getattr(process, "alpaka"+name)
        mod.alpaka = cms.untracked.PSet(backend = cms.untracked.string(args.moduleBackend))
if args.expectBackend == "cuda_async":
    process.alpakaGlobalConsumer.expectSize = 20
    process.alpakaStreamConsumer.expectSize = 25
    process.alpakaStreamInstanceConsumer.expectSize = 36
    process.alpakaStreamSynchronizingConsumer.expectSize = 20

process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('testAlpaka.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_alpakaGlobalProducer_*_*',
        'keep *_alpakaStreamProducer_*_*',
        'keep *_alpakaStreamSynchronizingProducer_*_*',
  )
)

process.t = cms.Task(
    process.intProduct,
    process.alpakaGlobalProducer,
    process.alpakaStreamProducer,
    process.alpakaStreamInstanceProducer,
    process.alpakaStreamSynchronizingProducer
)
process.p = cms.Path(
    process.alpakaGlobalConsumer+
    process.alpakaStreamConsumer+
    process.alpakaStreamInstanceConsumer+
    process.alpakaStreamSynchronizingConsumer,
    process.t
)
process.ep = cms.EndPath(process.output)
