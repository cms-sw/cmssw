import os

import FWCore.ParameterSet.VarParsing as VarParsing


args = VarParsing.VarParsing("analysis")

args.register(
    "numberOfThreads",
    1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of CMSSW threads"
)

args.register(
    "numberOfStreams",
    1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of CMSSW streams"
)

args.register(
    "numberOfEvents",
    1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of events to process"
)

args.register(
    "backend",
    "serial_sync",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Accelerator backend: serial_sync, cuda_async, or rocm_async"
)

args.register(
    "batchSize",
    8,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,        
    "Batch size"
)

args.register(
    "environment",
    0,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,        
    "Environment: 0 - production, 1 - development, 2 - test, 3 - debug"
)

args.register(
    "simpleNet",
    "PhysicsTools/PyTorchAlpakaTest/data/SimpleNet.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "SimpleNet model (just-in-time compiled)"
)

args.register(
    "maskedNet",
    "PhysicsTools/PyTorchAlpakaTest/data/MaskedNet.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "MaskedNet model (just-in-time compiled)"
)

args.register(
    "multiHeadNet",
    "PhysicsTools/PyTorchAlpakaTest/data/MultiHeadNet.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "MultiHeadNet model (just-in-time compiled)"
)

args.register(
    "tinyResNet",
    "PhysicsTools/PyTorchAlpakaTest/data/TinyResNet.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "TinyResNet model (just-in-time compiled)"
)

args.register(
    "only",
    ['simpleNet', 'multiHeadNet', 'maskedNet', 'tinyResNet'],
    VarParsing.VarParsing.multiplicity.list,
    VarParsing.VarParsing.varType.string,         
    "Run selected test(s): simpleNet, maskedNet, multiHeadNet, tinyResNet. Default is all modules run in parallel."
)