import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test various Alpaka module types')

parser.add_argument("--cuda", help="Use CUDA backend", action="store_true")
parser.add_argument("--run", type=int, help="Run number (default: 1)", default=1)

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

# TODO: just a temporary mechanism until we get something better that
# works also for ES modules. Absolutely NOT for wider use.
def setToCUDA(m):
    m._TypedParameterizable__type = m._TypedParameterizable__type.replace("alpaka_serial_sync", "alpaka_cuda_async")

process = cms.Process('TEST')

process.source = cms.Source('EmptySource',
    firstRun = cms.untracked.uint32(args.run)
)

process.maxEvents.input = 10

process.load('Configuration.StandardSequences.Accelerators_cff')
process.AlpakaServiceSerialSync = cms.Service('AlpakaServiceSerialSync')
if args.cuda:
    process.AlpakaServiceSerialSync.enabled = cms.untracked.bool(False)
    process.AlpakaServiceCudaAsync = cms.Service('AlpakaServiceCudaAsync')

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

process.alpakaESProducerA = cms.ESProducer("alpaka_serial_sync::TestAlpakaESProducerA")
process.alpakaESProducerB = cms.ESProducer("alpaka_serial_sync::TestAlpakaESProducerB")
process.alpakaESProducerC = cms.ESProducer("alpaka_serial_sync::TestAlpakaESProducerC")
process.alpakaESProducerD = cms.ESProducer("alpaka_serial_sync::TestAlpakaESProducerD")
if args.cuda:
    setToCUDA(process.alpakaESProducerA)
    setToCUDA(process.alpakaESProducerB)
    setToCUDA(process.alpakaESProducerC)
    setToCUDA(process.alpakaESProducerD)

process.intProduct = cms.EDProducer("IntProducer", ivalue = cms.int32(42))

process.alpakaGlobalProducer = cms.EDProducer("alpaka_serial_sync::TestAlpakaGlobalProducer",
    size = cms.int32(10)
)
process.alpakaStreamProducer = cms.EDProducer("alpaka_serial_sync::TestAlpakaStreamProducer",
    source = cms.InputTag("intProduct"),
    size = cms.int32(5)
)
process.alpakaStreamSynchronizingProducer = cms.EDProducer("alpaka_serial_sync::TestAlpakaStreamSynchronizingProducer",
    source = cms.InputTag("alpakaGlobalProducer")
)
if args.cuda:
    setToCUDA(process.alpakaGlobalProducer)
    setToCUDA(process.alpakaStreamProducer)
    setToCUDA(process.alpakaStreamSynchronizingProducer)

process.alpakaGlobalConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaGlobalProducer")
)
process.alpakaStreamConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamProducer")
)
process.alpakaStreamSynchronizingConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamSynchronizingProducer")
)

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
