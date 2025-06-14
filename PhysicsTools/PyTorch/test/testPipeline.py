import FWCore.ParameterSet.Config as cms
from PhysicsTools.PyTorch.options import args
  
args.parseArguments()
process = cms.Process("TestPipeline")

# enable multithreading
process.options.numberOfThreads = args.numberOfThreads if args.numberOfThreads > 1 else 1 
process.options.numberOfStreams = args.numberOfStreams if args.numberOfStreams > 1 else 1 

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

# process a limited number of events
process.maxEvents.input = args.numberOfEvents if args.numberOfEvents > 1 else 1 

# empty source
process.source = cms.Source("EmptySource")

# print a message every event
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = False

# load pipeline chain
process.load("PhysicsTools.PyTorch.modules")

# setup chain configs
process.DataProducer = process.DataProducerStruct.clone(
    batchSize = cms.uint32(args.batchSize if args.batchSize > 1 else 1),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)

# JIT models
process.JitClassificationProducer = process.JitClassificationProducerStruct.clone(
    inputs = cms.InputTag('DataProducer'),
    modelPath = cms.FileInPath(args.classificationModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
process.JitClassificationProducerCpu = process.JitClassificationProducerStruct.clone(
    inputs = cms.InputTag('DataProducer'),
    modelPath = cms.FileInPath(args.classificationModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("serial_sync")  # force serial backend to emulate heterogeneous execution
    ),
)
process.JitRegressionProducer = process.JitRegressionProducerStruct.clone(
    inputs = cms.InputTag('DataProducer'),
    modelPath = cms.FileInPath(args.regressionModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
process.JitRegressionProducerCpu = process.JitRegressionProducerStruct.clone(
    inputs = cms.InputTag('DataProducer'),
    modelPath = cms.FileInPath(args.regressionModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("serial_sync")  # force serial backend to emulate heterogeneous execution
    ),
)

# AOT models
regressionModelPath = args.regressionModelPathCpu
if args.backend == "cuda_async":
    regressionModelPath = args.regressionModelPathCuda
process.AotRegressionProducer = process.AotRegressionProducerStruct.clone(
    inputs = cms.InputTag('DataProducer'),
    modelPath = cms.FileInPath(regressionModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)
classificationModelPath = args.classificationModelPathCpu
if args.backend == "cuda_async":
    classificationModelPath = args.classificationModelPathCuda
process.AotClassificationProducer = process.AotClassificationProducerStruct.clone(
    inputs = cms.InputTag('DataProducer'),
    modelPath = cms.FileInPath(classificationModelPath),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)

process.CombinatoricsProducer = process.CombinatoricsProducerStruct.clone(
    inputs = cms.InputTag('DataProducer'),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string(args.backend)
    ),
)

# schedule the modules
process.path = cms.Path(
    process.DataProducer +
    process.JitClassificationProducer +
    # process.JitClassificationProducerCpu +
    # process.JitRegressionProducer +
    process.JitRegressionProducerCpu +

    # disable since no proper support is available yet
    # process.AotRegressionProducer +
    # process.AotClassificationProducer +

    process.CombinatoricsProducer
)

process.schedule = cms.Schedule(
    process.path
)
