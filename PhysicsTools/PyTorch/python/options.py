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
    10,
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
    2**20,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,        
    "Batch size"
)

args.register(
    "classificationModelPath",
    "PhysicsTools/PyTorch/models/jit_classification_model.pt",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Classification model (just-in-time compiled)"
)

args.register(
    "regressionModelPathCpu",
    "PhysicsTools/PyTorch/models/aot_regression_model_cpu_el9_amd64_gcc12.pt2",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Regression model (ahead-of-time compiled - shared library)"
)

args.register(
    "regressionModelPathCuda",
    "PhysicsTools/PyTorch/models/aot_regression_model_cuda_el9_amd64_gcc12.pt2",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,         
    "Regression model (ahead-of-time compiled - shared library wth CUDA support)"
)
