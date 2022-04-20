from FWCore.ParameterSet.VarParsing import VarParsing
import FWCore.ParameterSet.Config as cms
import os, sys, json

# module/model correspondence
models = {
    "TritonImageProducer": ["inception_graphdef", "densenet_onnx"],
    "TritonGraphProducer": ["gat_test"],
    "TritonGraphFilter": ["gat_test"],
    "TritonGraphAnalyzer": ["gat_test"],
    "TritonIdentityProducer": ["ragged_io"],
}

# other choices
allowed_modes = ["Async","PseudoAsync","Sync"]
allowed_compression = ["none","deflate","gzip"]
allowed_devices = ["auto","cpu","gpu"]

options = VarParsing()
options.register("maxEvents", -1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Number of events to process (-1 for all)")
options.register("serverName", "default", VarParsing.multiplicity.singleton, VarParsing.varType.string, "name for server (used internally)")
options.register("address", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "server address")
options.register("port", 8001, VarParsing.multiplicity.singleton, VarParsing.varType.int, "server port")
options.register("timeout", 30, VarParsing.multiplicity.singleton, VarParsing.varType.int, "timeout for requests")
options.register("params", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "json file containing server address/port")
options.register("threads", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "number of threads")
options.register("streams", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int, "number of streams")
options.register("modules", "TritonGraphProducer", VarParsing.multiplicity.list, VarParsing.varType.string, "list of modules to run (choices: {})".format(', '.join(models)))
options.register("models","gat_test", VarParsing.multiplicity.list, VarParsing.varType.string, "list of models (same length as modules, or just 1 entry if all modules use same model)")
options.register("mode","Async", VarParsing.multiplicity.singleton, VarParsing.varType.string, "mode for client (choices: {})".format(', '.join(allowed_modes)))
options.register("verbose", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "enable verbose output")
options.register("brief", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "briefer output for graph modules")
options.register("fallbackName", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "name for fallback server")
options.register("unittest", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "unit test mode: reduce input sizes")
options.register("testother", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "also test gRPC communication if shared memory enabled, or vice versa")
options.register("shm", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "enable shared memory")
options.register("compression", "", VarParsing.multiplicity.singleton, VarParsing.varType.string, "enable I/O compression (choices: {})".format(', '.join(allowed_compression)))
options.register("ssl", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "enable SSL authentication for server communication")
options.register("device","auto", VarParsing.multiplicity.singleton, VarParsing.varType.string, "specify device for fallback server (choices: {})".format(', '.join(allowed_devices)))
options.register("docker", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "use Docker for fallback server")
options.register("tries", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int, "number of retries for failed request")
options.parseArguments()

if len(options.params)>0:
    with open(options.params,'r') as pfile:
        pdict = json.load(pfile)
    options.address = pdict["address"]
    options.port = int(pdict["port"])
    print("server = "+options.address+":"+str(options.port))

# check models and modules
if len(options.modules)!=len(options.models):
    # assigning to VarParsing.multiplicity.list actually appends to existing value(s)
    if len(options.models)==1: options.models = [options.models[0]]*(len(options.modules)-1)
    else: raise ValueError("Arguments for modules and models must have same length")
for im,module in enumerate(options.modules):
    if module not in models:
        raise ValueError("Unknown module: {}".format(module))
    model = options.models[im]
    if model not in models[module]:
        raise ValueError("Unsupported model {} for module {}".format(model,module))

# check modes
if options.mode not in allowed_modes:
    raise ValueError("Unknown mode: {}".format(options.mode))

# check compression
if len(options.compression)>0 and options.compression not in allowed_compression:
    raise ValueError("Unknown compression setting: {}".format(options.compression))

# check devices
options.device = options.device.lower()
if options.device not in allowed_devices:
	raise ValueError("Unknown device: {}".format(options.device))

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
process = cms.Process('tritonTest',enableSonicTriton)

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("EmptySource")

process.TritonService.verbose = options.verbose
process.TritonService.fallback.verbose = options.verbose
process.TritonService.fallback.useDocker = options.docker
if len(options.fallbackName)>0:
    process.TritonService.fallback.instanceBaseName = options.fallbackName
if options.device != "auto":
    process.TritonService.fallback.useGPU = options.device=="gpu"
if len(options.address)>0:
    process.TritonService.servers.append(
        cms.PSet(
            name = cms.untracked.string(options.serverName),
            address = cms.untracked.string(options.address),
            port = cms.untracked.uint32(options.port),
            useSsl = cms.untracked.bool(options.ssl),
            rootCertificates = cms.untracked.string(""),
            privateKey = cms.untracked.string(""),
            certificateChain = cms.untracked.string(""),
        )
    )

# Let it run
process.p = cms.Path()

modules = {
    "Producer": cms.EDProducer,
    "Filter": cms.EDFilter,
    "Analyzer": cms.EDAnalyzer,
}

keepMsgs = ['TritonClient','TritonService']

for im,module in enumerate(options.modules):
    model = options.models[im]
    Module = [obj for name,obj in modules.items() if name in module][0]
    setattr(process, module,
        Module(module,
            Client = cms.PSet(
                mode = cms.string(options.mode),
                preferredServer = cms.untracked.string(""),
                timeout = cms.untracked.uint32(options.timeout),
                modelName = cms.string(model),
                modelVersion = cms.string(""),
                modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/{}/config.pbtxt".format(model)),
                verbose = cms.untracked.bool(options.verbose),
                allowedTries = cms.untracked.uint32(options.tries),
                useSharedMemory = cms.untracked.bool(options.shm),
                compression = cms.untracked.string(options.compression),
            )
        )
    )
    processModule = getattr(process, module)
    if module=="TritonImageProducer":
        processModule.batchSize = cms.int32(1)
        processModule.topN = cms.uint32(5)
        processModule.imageList = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/{}/{}_labels.txt".format(model,model.split('_')[0]))
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
        keepMsgs.extend([_module2,_module2+':TritonClient'])

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

