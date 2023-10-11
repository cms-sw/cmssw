import FWCore.ParameterSet.Config as cms
import sys
import argparse

# This configuration demonstrates how to run an EDProducer on two
# possibly different backends: one is the "portable" and another is
# explicitly a host backend, and how to handle (one model of)
# ESProducer in such case.

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test various Alpaka module types')

parser.add_argument("--expectBackend", type=str, help="Expect this backend to run")
parser.add_argument("--run", type=int, help="Run number (default: 1)", default=1)

args = parser.parse_args()

process = cms.Process('TEST')

process.source = cms.Source('EmptySource',
    firstRun = cms.untracked.uint32(args.run)
)

process.maxEvents.input = 10

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.alpakaESRecordASource = cms.ESSource("EmptyESSource",
    recordName = cms.string('AlpakaESTestRecordA'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.esProducerA = cms.ESProducer("cms::alpakatest::TestESProducerA", value = cms.int32(42))

process.alpakaESProducerA = cms.ESProducer("TestAlpakaESProducerA@alpaka")


process.producer = cms.EDProducer("TestAlpakaGlobalProducerOffset@alpaka",
    xvalue = cms.PSet(
        alpaka_serial_sync = cms.double(1.0),
        alpaka_cuda_async = cms.double(2.0)
    )
)
process.producerHost = process.producer.clone(
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("serial_sync")
    )
)

process.compare = cms.EDAnalyzer("TestAlpakaHostDeviceCompare",
    srcHost = cms.untracked.InputTag("producerHost"),
    srcDevice = cms.untracked.InputTag("producer"),
    expectedXdiff = cms.untracked.double(0.0)
)
if args.expectBackend == "cuda_async":
    process.compare.expectedXdiff = -1.0

process.t = cms.Task(process.producer, process.producerHost)
process.p = cms.Path(process.compare, process.t)
