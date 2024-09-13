import FWCore.ParameterSet.Config as cms
import os, sys, json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
allowed_containers = ["apptainer","docker","podman","podman-hpc"]

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--maxEvents", default=-1, type=int, help="Number of events to process (-1 for all)")
parser.add_argument("--serverName", default="default", type=str, help="name for server (used internally)")
parser.add_argument("--address", default="", type=str, help="server address")
parser.add_argument("--port", default=8001, type=int, help="server port")
parser.add_argument("--timeout", default=30, type=int, help="timeout for requests")
parser.add_argument("--timeoutUnit", default="seconds", type=str, help="unit for timeout")
parser.add_argument("--params", default="", type=str, help="json file containing server address/port")
parser.add_argument("--threads", default=1, type=int, help="number of threads")
parser.add_argument("--streams", default=0, type=int, help="number of streams")
parser.add_argument("--modules", metavar=("MODULES"), default=["TritonGraphProducer"], nargs='+', type=str, choices=list(models), help="list of modules to run (choices: %(choices)s)")
parser.add_argument("--models", default=["gat_test"], nargs='+', type=str, help="list of models (same length as modules, or just 1 entry if all modules use same model)")
parser.add_argument("--mode", default="Async", type=str, choices=allowed_modes, help="mode for client")
parser.add_argument("--verbose", default=False, action="store_true", help="enable all verbose output")
parser.add_argument("--verboseClient", default=False, action="store_true", help="enable verbose output for clients")
parser.add_argument("--verboseServer", default=False, action="store_true", help="enable verbose output for server")
parser.add_argument("--verboseService", default=False, action="store_true", help="enable verbose output for TritonService")
parser.add_argument("--verboseDiscovery", default=False, action="store_true", help="enable verbose output just for server discovery in TritonService")
parser.add_argument("--brief", default=False, action="store_true", help="briefer output for graph modules")
parser.add_argument("--fallbackName", default="", type=str, help="name for fallback server")
parser.add_argument("--unittest", default=False, action="store_true", help="unit test mode: reduce input sizes")
parser.add_argument("--testother", default=False, action="store_true", help="also test gRPC communication if shared memory enabled, or vice versa")
parser.add_argument("--noShm", default=False, action="store_true", help="disable shared memory")
parser.add_argument("--compression", default="", type=str, choices=allowed_compression, help="enable I/O compression")
parser.add_argument("--ssl", default=False, action="store_true", help="enable SSL authentication for server communication")
parser.add_argument("--device", default="auto", type=str.lower, choices=allowed_devices, help="specify device for fallback server")
parser.add_argument("--container", default="apptainer", type=str.lower, choices=allowed_containers, help="specify container for fallback server")
parser.add_argument("--tries", default=0, type=int, help="number of retries for failed request")
options = parser.parse_args()

if len(options.params)>0:
    with open(options.params,'r') as pfile:
        pdict = json.load(pfile)
    options.address = pdict["address"]
    options.port = int(pdict["port"])
    print("server = "+options.address+":"+str(options.port))

# check models and modules
if len(options.modules)!=len(options.models):
    # assigning to VarParsing.multiplicity.list actually appends to existing value(s)
    if len(options.models)==1: options.models = [options.models[0]]*(len(options.modules))
    else: raise ValueError("Arguments for modules and models must have same length")
for im,module in enumerate(options.modules):
    model = options.models[im]
    if model not in models[module]:
        raise ValueError("Unsupported model {} for module {}".format(model,module))

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
process = cms.Process('tritonTest',enableSonicTriton)

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("EmptySource")

process.TritonService.verbose = options.verbose or options.verboseService or options.verboseDiscovery
process.TritonService.fallback.verbose = options.verbose or options.verboseServer
process.TritonService.fallback.container = options.container
process.TritonService.fallback.device = options.device
if len(options.fallbackName)>0:
    process.TritonService.fallback.instanceBaseName = options.fallbackName
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

keepMsgs = []
if options.verbose or options.verboseDiscovery:
    keepMsgs.append('TritonDiscovery')
if options.verbose or options.verboseClient:
    keepMsgs.append('TritonClient')
if options.verbose or options.verboseService:
    keepMsgs.append('TritonService')

for im,module in enumerate(options.modules):
    model = options.models[im]
    Module = [obj for name,obj in modules.items() if name in module][0]
    setattr(process, module,
        Module(module,
            Client = cms.PSet(
                mode = cms.string(options.mode),
                preferredServer = cms.untracked.string(""),
                timeout = cms.untracked.uint32(options.timeout),
                timeoutUnit = cms.untracked.string(options.timeoutUnit),
                modelName = cms.string(model),
                modelVersion = cms.string(""),
                modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/{}/config.pbtxt".format(model)),
                verbose = cms.untracked.bool(options.verbose or options.verboseClient),
                allowedTries = cms.untracked.uint32(options.tries),
                useSharedMemory = cms.untracked.bool(not options.noShm),
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
    if options.verbose or options.verboseClient:
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
        if options.verbose or options.verboseClient:
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

