import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test various Alpaka module types')

parser.add_argument("--accelerators", type=str, help="Set process.options.accelerators (comma-separated string, default is to use default)", default="")
parser.add_argument("--moduleBackend", type=str, help="Set Alpaka backend for module instances", default="")
parser.add_argument("--run", type=int, help="Run number (default: 1)", default=1)

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

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

process.alpakaESProducerA = cms.ESProducer("TestAlpakaESProducerA@alpaka")
process.alpakaESProducerB = cms.ESProducer("TestAlpakaESProducerB@alpaka")
process.alpakaESProducerC = cms.ESProducer("TestAlpakaESProducerC@alpaka")
process.alpakaESProducerD = cms.ESProducer("TestAlpakaESProducerD@alpaka")

process.intProduct = cms.EDProducer("IntProducer", ivalue = cms.int32(42))

process.alpakaGlobalProducer = cms.EDProducer("TestAlpakaGlobalProducer@alpaka",
    size = cms.int32(10)
)
process.alpakaStreamProducer = cms.EDProducer("TestAlpakaStreamProducer@alpaka",
    source = cms.InputTag("intProduct"),
    size = cms.int32(5)
)
process.alpakaStreamSynchronizingProducer = cms.EDProducer("TestAlpakaStreamSynchronizingProducer@alpaka",
    source = cms.InputTag("alpakaGlobalProducer")
)

process.alpakaGlobalConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaGlobalProducer")
)
process.alpakaStreamConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamProducer")
)
process.alpakaStreamSynchronizingConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamSynchronizingProducer")
)

if args.moduleBackend != "":
    for name in ["ESProducerA", "ESProducerB", "ESProducerC", "ESProducerD",
                 "GlobalProducer", "StreamProducer", "StreamSynchronizingProducer"]:
        mod = getattr(process, "alpaka"+name)
        mod.alpaka = cms.untracked.PSet(backend = cms.untracked.string(args.moduleBackend))

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
    process.alpakaStreamSynchronizingProducer
)
process.p = cms.Path(
    process.alpakaGlobalConsumer+
    process.alpakaStreamConsumer+
    process.alpakaStreamSynchronizingConsumer,
    process.t
)
process.ep = cms.EndPath(process.output)
