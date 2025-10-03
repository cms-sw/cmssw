import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorchAlpakaTest.options_cff import args
from PhysicsTools.PyTorchAlpakaTest.modules import (
    torchtest_DataSource_alpaka,
    torchtest_SimpleNet_alpaka,
    torchtest_MultiHeadNet_alpaka,
    torchtest_MaskedNet_alpaka,
    torchtest_TinyResNet_alpaka,
    torchtest_InspectionSink
)
  
args.parseArguments()
process = cms.Process("testPyTorchAlpakaHeterogeneousPipeline")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 

# logging
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.PyTorchService = {}

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")
process.PyTorchService = cms.Service("PyTorchService")

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source
process.source = cms.Source("EmptySource")

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options.wantSummary = True

# setup chain configs
process.DataSource = torchtest_DataSource_alpaka(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    environment = cms.untracked.int32(args.environment)
)
process.SimpleNet = torchtest_SimpleNet_alpaka(
    model = cms.FileInPath(args.simpleNet),
    particles = 'DataSource',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("serial_sync")  # force serial backend to emulate heterogeneous pipeline
    ),
    environment = cms.untracked.int32(args.environment)
)
process.MultiHeadNet = torchtest_MultiHeadNet_alpaka(
    model = cms.FileInPath(args.multiHeadNet),
    particles = 'DataSource',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    environment = cms.untracked.int32(args.environment)
)
process.MaskedNet = torchtest_MaskedNet_alpaka(
    model = cms.FileInPath(args.maskedNet),
    particles = 'DataSource',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    environment = cms.untracked.int32(args.environment)
)
process.TinyResNet = torchtest_TinyResNet_alpaka(
    model = cms.FileInPath(args.tinyResNet),
    images = 'DataSource',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
    environment = cms.untracked.int32(args.environment)
)
process.InspectionSink = torchtest_InspectionSink(
    particles = 'DataSource',
    simple_net = 'SimpleNet',
    masked_net = 'MaskedNet',
    multi_head_net = 'MultiHeadNet',
    images = 'DataSource',
    resnet18 = 'TinyResNet',
    environment = cms.untracked.int32(args.environment)
)


# schedule the modules
process.path = cms.Path(
    process.DataSource + 
    process.SimpleNet +
    process.MultiHeadNet +
    process.MaskedNet +
    process.TinyResNet +
    process.InspectionSink
)