from FWCore.ParameterSet.VarParsing import VarParsing
import FWCore.ParameterSet.Config as cms
import os, sys, json, six

options = VarParsing("analysis")
options.register("serverName", "default", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("address", "", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("port", 8001, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("timeout", 30, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("params", "", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("threads", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("streams", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("modules", "TritonImageProducer", VarParsing.multiplicity.list, VarParsing.varType.string)
options.register("modelName","resnet50_netdef", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("mode","Async", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("verbose", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register("brief", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register("unittest", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register("testother", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register("shm", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register("device","auto", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("docker", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register("tries", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.parseArguments()

if len(options.params)>0:
    with open(options.params,'r') as pfile:
        pdict = json.load(pfile)
    options.address = pdict["address"]
    options.port = int(pdict["port"])
    print("server = "+options.address+":"+str(options.port))

# check devices
options.device = options.device.lower()
allowed_devices = ["auto","cpu","gpu"]
if options.device not in allowed_devices:
	raise ValueError("Unknown device: "+options.device)

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
process = cms.Process('tritonTest',enableSonicTriton)

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("EmptySource")

process.TritonService.verbose = options.verbose
process.TritonService.fallback.verbose = options.verbose
process.TritonService.fallback.useDocker = options.docker
if options.device != "auto":
    process.TritonService.fallback.useGPU = options.device=="gpu"
if len(options.address)>0:
    process.TritonService.servers.append(
        cms.PSet(
            name = cms.untracked.string(options.serverName),
            address = cms.untracked.string(options.address),
            port = cms.untracked.uint32(options.port),
        )
    )

# Let it run
process.p = cms.Path()

# check module/model
models = {
    "TritonImageProducer": "resnet50_netdef",
    "TritonGraphProducer": "gat_test",
    "TritonGraphFilter": "gat_test",
    "TritonGraphAnalyzer": "gat_test",
}

modules = {
    "Producer": cms.EDProducer,
    "Filter": cms.EDFilter,
    "Analyzer": cms.EDAnalyzer,
}

keepMsgs = ['TritonClient','TritonService']
for module in options.modules:
    if module not in models:
        raise ValueError("Unknown module: "+module)
    Module = [obj for name,obj in six.iteritems(modules) if name in module][0]
    setattr(process, module,
        Module(module,
            Client = cms.PSet(
                mode = cms.string(options.mode),
                preferredServer = cms.untracked.string(""),
                timeout = cms.untracked.uint32(options.timeout),
                modelName = cms.string(models[module]),
                modelVersion = cms.string(""),
                modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/{}/config.pbtxt".format(models[module])),
                verbose = cms.untracked.bool(options.verbose),
                allowedTries = cms.untracked.uint32(options.tries),
                useSharedMemory = cms.untracked.bool(options.shm),
            )
        )
    )
    processModule = getattr(process, module)
    if module=="TritonImageProducer":
        processModule.batchSize = cms.int32(1)
        processModule.topN = cms.uint32(5)
        processModule.imageList = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/resnet50_netdef/resnet50_labels.txt")
    elif "TritonGraph" in module:
        if options.unittest:
            # reduce input size for unit test
            processModule.nodeMin = cms.uint32(1)
            processModule.nodeMax = cms.uint32(10)
            processModule.edgeMin = cms.uint32(20)
            processModule.edgeMax = cms.uint32(40)
        else:
            processModule.nodeMin = cms.uint32(100)
            processModule.nodeMax = cms.uint32(4000)
            processModule.edgeMin = cms.uint32(8000)
            processModule.edgeMax = cms.uint32(15000)
        processModule.brief = cms.bool(options.brief)
    process.p += processModule
    keepMsgs.extend([module,module+':TritonClient'])
    if options.testother:
        # clone modules to test both gRPC and shared memory
        _module2 = module+"GRPC" if processModule.Client.useSharedMemory else "SHM"
        setattr(process, _module2,
            processModule.clone(
                Client = dict(useSharedMemory = not processModule.Client.useSharedMemory)
            )
        )
        processModule2 = getattr(process, _module2)
        process.p += processModule2

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 500
for msg in keepMsgs:
    setattr(process.MessageLogger.cerr,msg,
        cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
        )
    )

if options.threads>0:
    process.options.numberOfThreads = options.threads
    process.options.numberOfStreams = options.streams

