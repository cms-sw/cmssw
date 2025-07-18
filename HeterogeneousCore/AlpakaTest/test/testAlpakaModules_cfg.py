import FWCore.ParameterSet.Config as cms
import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test various Alpaka module types')

parser.add_argument("--accelerators", type=str, help="Set process.options.accelerators (comma-separated string, default is to use default)", default="")
parser.add_argument("--moduleBackend", type=str, help="Set Alpaka backend via module instances", default="")
parser.add_argument("--processAcceleratorBackend", type=str, help="Set Alpaka backend via ProcessAcceleratorAlpaka", default="")
parser.add_argument("--expectBackend", type=str, help="Expect this backend to run")
parser.add_argument("--moduleSynchronize", action="store_true", help="Set synchronize parameter via module instances", default="")
parser.add_argument("--processAcceleratorSynchronize", action="store_true", help="Set synchronize parameter via ProcessAcceleratorAlpaka", default="")
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
process.alpakaESProducerE = cms.ESProducer("TestAlpakaESProducerE@alpaka")
process.alpakaESProducerNull = cms.ESProducer("TestAlpakaESProducerNull@alpaka",
    appendToDataLabel = cms.string("null"),
)

# PortableMultiCollection
from HeterogeneousCore.AlpakaTest.testAlpakaESProducerAMulti_cfi import testAlpakaESProducerAMulti 

process.intProduct = cms.EDProducer("IntProducer", ivalue = cms.int32(42))
process.alpakaESProducerAMulti = testAlpakaESProducerAMulti.clone(appendToDataLabel = cms.string("appendedLabel"))

from HeterogeneousCore.AlpakaTest.testAlpakaGlobalProducer_cfi import testAlpakaGlobalProducer
process.alpakaGlobalProducer = testAlpakaGlobalProducer.clone(
    eventSetupSource = cms.ESInputTag("alpakaESProducerA", "appendedLabel"),
    eventSetupSourceMulti = cms.ESInputTag("alpakaESProducerAMulti", "appendedLabel"),
    size = dict(
        alpaka_serial_sync = 10,
        alpaka_cuda_async = 20,
        alpaka_rocm_async = 30,
    )
)
process.alpakaGlobalProducerE = cms.EDProducer("TestAlpakaGlobalProducerE@alpaka",
    source = cms.InputTag("alpakaGlobalProducer")
)
process.alpakaGlobalProducerCopyToDeviceCache = cms.EDProducer("TestAlpakaGlobalProducerCopyToDeviceCache@alpaka",
    source = cms.InputTag("alpakaGlobalProducer"),
    x = cms.int32(3),
    y = cms.int32(4),
    z = cms.int32(5),
)
process.alpakaGlobalProducerMoveToDeviceCache = cms.EDProducer("TestAlpakaGlobalProducerMoveToDeviceCache@alpaka",
    source = cms.InputTag("alpakaGlobalProducer"),
    x = cms.int32(32),
    y = cms.int32(42),
    z = cms.int32(52),
)
process.alpakaGlobalProducerImplicitCopyToDevice = cms.EDProducer("TestAlpakaGlobalProducerImplicitCopyToDevice@alpaka")
process.alpakaStreamProducer = cms.EDProducer("TestAlpakaStreamProducer@alpaka",
    source = cms.InputTag("intProduct"),
    eventSetupSource = cms.ESInputTag("alpakaESProducerB", "explicitLabel"),
    size = cms.PSet(
        alpaka_serial_sync = cms.int32(5),
        alpaka_cuda_async = cms.int32(25),
        alpaka_rocm_async = cms.int32(125),
    )
)
process.alpakaStreamInstanceProducer = cms.EDProducer("TestAlpakaStreamProducer@alpaka",
    source = cms.InputTag("intProduct"),
    eventSetupSource = cms.ESInputTag("alpakaESProducerB", "explicitLabel"),
    productInstanceName = cms.string("testInstance"),
    size = cms.PSet(
        alpaka_serial_sync = cms.int32(6),
        alpaka_cuda_async = cms.int32(36),
        alpaka_rocm_async = cms.int32(216),
    )
)
process.alpakaStreamSynchronizingProducer = cms.EDProducer("TestAlpakaStreamSynchronizingProducer@alpaka",
    source = cms.InputTag("alpakaGlobalProducer"),
    intSource = cms.InputTag("intProduct"),
    expectedInt = cms.int32(84) # sum of intProduct and esProducerA
)
process.alpakaStreamSynchronizingProducerToDevice = cms.EDProducer("TestAlpakaStreamSynchronizingProducerToDevice@alpaka",
    size = cms.PSet(
        alpaka_serial_sync = cms.int32(1),
        alpaka_cuda_async = cms.int32(2),
        alpaka_rocm_async = cms.int32(3),
    )
)

process.alpakaGlobalConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaGlobalProducer"),
    expectSize = cms.int32(10),
    expectBackend = cms.string("SerialSync")
)
process.alpakaGlobalDeviceConsumer = cms.EDProducer("TestAlpakaGlobalProducerNoOutput@alpaka",
    source = cms.InputTag("alpakaGlobalProducer")
)
process.alpakaGlobalConsumerE = process.alpakaGlobalConsumer.clone(
    source = "alpakaGlobalProducerE",
    expectXvalues = cms.vdouble([(i%2)*10+1 + abs(27)+i*2 for i in range(0,5)] + [0]*5)
)
process.alpakaGlobalConsumerCopyToDeviceCache = process.alpakaGlobalConsumer.clone(
    source = "alpakaGlobalProducerCopyToDeviceCache",
    expectXvalues = cms.vdouble([3]*10)
)
process.alpakaGlobalConsumerMoveToDeviceCache = process.alpakaGlobalConsumer.clone(
    source = "alpakaGlobalProducerMoveToDeviceCache",
    expectXvalues = cms.vdouble([32]*10)
)
from HeterogeneousCore.AlpakaTest.modules import TestAlpakaVerifyObjectOnDevice_alpaka
process.alpakaGlobalConsumerImplicitCopyToDevice = TestAlpakaVerifyObjectOnDevice_alpaka(
    source = "alpakaGlobalProducerImplicitCopyToDevice"
)
process.alpakaGlobalConsumerImplicitCopyToDeviceInstance = TestAlpakaVerifyObjectOnDevice_alpaka(
    source = ("alpakaGlobalProducerImplicitCopyToDevice", "instance")
)
process.alpakaStreamConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamProducer"),
    expectSize = cms.int32(5),
    expectBackend = cms.string("SerialSync")
)
process.alpakaStreamDeviceConsumer = process.alpakaGlobalDeviceConsumer.clone(
    source = "alpakaStreamProducer"
)
process.alpakaStreamInstanceConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamInstanceProducer", "testInstance"),
    expectSize = cms.int32(6),
    expectBackend = cms.string("SerialSync")
)
process.alpakaStreamSynchronizingConsumer = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("alpakaStreamSynchronizingProducer"),
    expectSize = cms.int32(10),
    expectBackend = cms.string("SerialSync")
)
process.alpakaStreamSynchronizingProducerToDeviceDeviceConsumer1 = process.alpakaGlobalDeviceConsumer.clone(
    source = "alpakaStreamSynchronizingProducerToDevice"
)
process.alpakaStreamSynchronizingProducerToDeviceDeviceConsumer2 = process.alpakaStreamSynchronizingProducerToDeviceDeviceConsumer1.clone()
process.alpakaNullESConsumer = cms.EDProducer("TestAlpakaGlobalProducerNullES@alpaka",
    eventSetupSource = cms.ESInputTag("", "null")
)

_postfixes = ["ESProducerA", "ESProducerB", "ESProducerC", "ESProducerD", "ESProducerE", "ESProducerAMulti",
              "ESProducerNull",
              "GlobalProducer", "GlobalProducerE",
              "GlobalProducerCopyToDeviceCache", "GlobalProducerMoveToDeviceCache",
              "GlobalProducerImplicitCopyToDevice",
              "StreamProducer", "StreamInstanceProducer",
              "StreamSynchronizingProducer", "StreamSynchronizingProducerToDevice",
              "GlobalConsumerImplicitCopyToDevice", "GlobalConsumerImplicitCopyToDeviceInstance",
              "GlobalDeviceConsumer", "StreamDeviceConsumer",
              "StreamSynchronizingProducerToDeviceDeviceConsumer1", "StreamSynchronizingProducerToDeviceDeviceConsumer2",
              "NullESConsumer"]
alpakaModules = ["alpaka"+x for x in _postfixes]
if args.processAcceleratorBackend != "":
    process.ProcessAcceleratorAlpaka.setBackend(args.processAcceleratorBackend)
if args.moduleBackend != "":
    for name in alpakaModules:
        getattr(process, name).alpaka = cms.untracked.PSet(backend = cms.untracked.string(args.moduleBackend))
if args.expectBackend == "cuda_async":
    def setExpect(m, size):
        m.expectSize = size
        m.expectBackend = "CudaAsync"
    setExpect(process.alpakaGlobalConsumer, size=20)
    setExpect(process.alpakaGlobalConsumerE, size=20)
    process.alpakaGlobalConsumerE.expectXvalues.extend([0]*(20-10))
    setExpect(process.alpakaGlobalConsumerCopyToDeviceCache, size=20)
    process.alpakaGlobalConsumerCopyToDeviceCache.expectXvalues = [3]*20
    setExpect(process.alpakaGlobalConsumerMoveToDeviceCache, size=20)
    process.alpakaGlobalConsumerMoveToDeviceCache.expectXvalues = [32]*20
    setExpect(process.alpakaStreamConsumer, size=25)
    setExpect(process.alpakaStreamInstanceConsumer, size=36)
    setExpect(process.alpakaStreamSynchronizingConsumer, size=20)
elif args.expectBackend == "rocm_async":
    def setExpect(m, size):
        m.expectSize = size
        m.expectBackend = "ROCmAsync"
    setExpect(process.alpakaGlobalConsumer, size = 30)
    setExpect(process.alpakaGlobalConsumerE, size = 30)
    process.alpakaGlobalConsumerE.expectXvalues.extend([0]*(30-10))
    setExpect(process.alpakaGlobalConsumerCopyToDeviceCache, size = 30)
    process.alpakaGlobalConsumerCopyToDeviceCache.expectXvalues = [3]*30
    setExpect(process.alpakaGlobalConsumerMoveToDeviceCache, size = 30)
    process.alpakaGlobalConsumerMoveToDeviceCache.expectXvalues = [32]*30
    setExpect(process.alpakaStreamConsumer, size = 125)
    setExpect(process.alpakaStreamInstanceConsumer, size = 216)
    setExpect(process.alpakaStreamSynchronizingConsumer, size = 30)

if args.processAcceleratorSynchronize:
    process.ProcessAcceleratorAlpaka.setSynchronize(True)
if args.moduleSynchronize:
    for name in alpakaModules:
        mod = getattr(process, name)
        if hasattr(mod, "alpaka"):
            mod.alpaka = dict(synchronize = cms.untracked.bool(True))
        else:
            mod.alpaka = cms.untracked.PSet(synchronize = cms.untracked.bool(True))

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
    process.alpakaGlobalProducerE,
    process.alpakaGlobalProducerCopyToDeviceCache,
    process.alpakaGlobalProducerMoveToDeviceCache,
    process.alpakaGlobalProducerImplicitCopyToDevice,
    process.alpakaStreamProducer,
    process.alpakaStreamInstanceProducer,
    process.alpakaStreamSynchronizingProducer,
    process.alpakaStreamSynchronizingProducerToDevice
)
process.p = cms.Path(
    process.alpakaGlobalConsumer+
    process.alpakaGlobalDeviceConsumer+
    process.alpakaGlobalConsumerE+
    process.alpakaGlobalConsumerCopyToDeviceCache+
    process.alpakaGlobalConsumerMoveToDeviceCache+
    process.alpakaGlobalConsumerImplicitCopyToDevice+
    process.alpakaGlobalConsumerImplicitCopyToDeviceInstance+
    process.alpakaStreamConsumer+
    process.alpakaStreamDeviceConsumer+
    process.alpakaStreamInstanceConsumer+
    process.alpakaStreamSynchronizingConsumer+
    process.alpakaStreamSynchronizingProducerToDeviceDeviceConsumer1+
    process.alpakaStreamSynchronizingProducerToDeviceDeviceConsumer2+
    process.alpakaNullESConsumer,
    process.t
)
process.ep = cms.EndPath(process.output)
