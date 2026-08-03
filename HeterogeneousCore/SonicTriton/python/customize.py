import FWCore.ParameterSet.Config as cms
import json

def getDefaultClientPSet():
    from HeterogeneousCore.SonicTriton.TritonGraphAnalyzer import TritonGraphAnalyzer
    temp = TritonGraphAnalyzer()
    return temp.Client

def getParser():
    allowed_compression = ["none","deflate","gzip"]
    allowed_devices = ["auto","cpu","gpu"]
    allowed_containers = ["apptainer","docker","podman","podman-hpc"]

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--maxEvents", default=-1, type=int, help="Number of events to process (-1 for all)")
    parser.add_argument("--address", nargs=3, action="append", metavar=("NAME", "HOST", "PORT"),
                        dest="addresses", default=[],
                        help="Triton server entry: name host port (repeatable, e.g. --address server1 0.0.0.0 8011 --address server2 0.0.0.0 8021)")
    parser.add_argument("--timeout", default=30, type=int, help="timeout for requests")
    parser.add_argument("--timeoutUnit", default="seconds", type=str, help="unit for timeout")
    parser.add_argument("--params", default="", type=str, help="json file containing server address/port(s) (single server dict, or list of server dicts[{'name': ..., 'address': ..., 'port': ...}])")
    parser.add_argument("--threads", default=1, type=int, help="number of threads")
    parser.add_argument("--streams", default=0, type=int, help="number of streams")
    parser.add_argument("--verbose", default=False, action="store_true", help="enable all verbose output")
    parser.add_argument("--verboseClient", default=False, action="store_true", help="enable verbose output for clients")
    parser.add_argument("--verboseServer", default=False, action="store_true", help="enable verbose output for server")
    parser.add_argument("--verboseService", default=False, action="store_true", help="enable verbose output for TritonService")
    parser.add_argument("--verboseDiscovery", default=False, action="store_true", help="enable verbose output just for server discovery in TritonService")
    parser.add_argument("--noShm", default=False, action="store_true", help="disable shared memory")
    parser.add_argument("--compression", default="", type=str, choices=allowed_compression, help="enable I/O compression")
    parser.add_argument("--ssl", default=False, action="store_true", help="enable SSL authentication for server communication")
    parser.add_argument("--tries", default=0, type=int, help="number of retries for failed request")
    parser.add_argument("--device", default="auto", type=str.lower, choices=allowed_devices, help="specify device for fallback server")
    parser.add_argument("--container", default="apptainer", type=str.lower, choices=allowed_containers, help="specify container for fallback server")
    parser.add_argument("--fallbackName", default="", type=str, help="name for fallback server")
    parser.add_argument("--imageName", default="", type=str, help="container image name for fallback server")
    parser.add_argument("--sandboxDir", default="", type=str, help="apptainer sandbox directory")
    parser.add_argument("--tempDir", default="", type=str, help="temp directory for fallback server")
    parser.add_argument("--retryAction", default="same", type=str, choices=["same","diff"], help="retry policy: same server or different server")

    return parser

def getOptions(parser, verbose=False):
    options = parser.parse_args()

    if len(options.params) > 0:
        with open(options.params, 'r') as pfile:
            pdict = json.load(pfile)

        server_list = pdict if isinstance(pdict, list) else [pdict]

        for entry in server_list:
            name = entry.get("name", "default")
            host = entry["address"]
            port = str(int(entry["port"]))
            options.addresses.append([name, host, port])
            if verbose:
                print("server (from params) = {}:{} [{}]".format(host, port, name))

    return options

def applyOptions(process, options, applyToModules=False):
    process.maxEvents.input = cms.untracked.int32(options.maxEvents)

    if options.threads>0:
        process.options.numberOfThreads = options.threads
        process.options.numberOfStreams = options.streams

    if options.verbose:
        configureLoggingAll(process)
    else:
        configureLogging(process,
            client=options.verboseClient,
            server=options.verboseServer,
            service=options.verboseService,
            discovery=options.verboseDiscovery
        )

    if hasattr(process,'TritonService'):
        process.TritonService.fallback.container = options.container
        process.TritonService.fallback.imageName = options.imageName
        process.TritonService.fallback.sandboxDir = options.sandboxDir
        process.TritonService.fallback.tempDir = options.tempDir
        process.TritonService.fallback.device = options.device
        if len(options.fallbackName)>0:
            process.TritonService.fallback.instanceBaseName = options.fallbackName
        for name, host, port in options.addresses:
            if options.verbose:
                print("server = {}:{} [{}]".format(host, port, name))
            process.TritonService.servers.append(
                dict(
                    name    = name,
                    address = host,
                    port    = int(port),
                    useSsl  = options.ssl,
                )
            )

    if applyToModules:
        process = configureModules(process, **getClientOptions(options))

    return process

def getClientOptions(options):
    action = cms.PSet(
                retryType = cms.string('RetrySameServerAction'),
                allowedTries = cms.untracked.uint32(options.tries))
    if options.retryAction != 'same':
        action.retryType = cms.string('RetryActionDiffServer')

    fallback = cms.PSet(retryType = cms.string('RetryFallbackServerAction'))
    return dict(
        compression = cms.untracked.string(options.compression),
        useSharedMemory = cms.untracked.bool(not options.noShm),
        timeout = cms.untracked.uint32(options.timeout),
        timeoutUnit = cms.untracked.string(options.timeoutUnit),
        Retry = cms.VPSet(action,fallback)
    )

def applyClientOptions(client, options):
    return configureClient(client, **getClientOptions(options))

def configureModules(process, modules=None, returnConfigured=False, **kwargs):
    if modules is None:
        modules = {}
        modules.update(process.producers_())
        modules.update(process.filters_())
        modules.update(process.analyzers_())
    configured = []
    for pname,producer in modules.items():
        if hasattr(producer,'Client'):
            producer.Client = configureClient(producer.Client, **kwargs)
            configured.append(pname)
    if returnConfigured:
        return process, configured
    else:
        return process

def configureClient(client, **kwargs):
    client.update_(kwargs)
    return client

def configureLogging(process, client=False, server=False, service=False, discovery=False):
    if not any([client, server, service, discovery]):
        return

    keepMsgs = []
    if discovery:
        keepMsgs.append('TritonDiscovery')
    if client:
        keepMsgs.append('TritonClient')
    if service:
        keepMsgs.append('TritonService')

    if hasattr(process,'TritonService'):
        process.TritonService.verbose = service or discovery
        process.TritonService.fallback.verbose = server
    if client:
        process, configured = configureModules(process, returnConfigured=True, verbose = True)
        for module in configured:
            keepMsgs.extend([module, module+':TritonClient'])

    process.MessageLogger.cerr.FwkReport.reportEvery = 500
    for msg in keepMsgs:
        setattr(process.MessageLogger.cerr, msg,
            dict(
                limit = 10000000,
            )
        )

    return process

# dedicated functions for cmsDriver customization

def configureLoggingClient(process):
    return configureLogging(process, client=True)

def configureLoggingServer(process):
    return configureLogging(process, server=True)

def configureLoggingService(process):
    return configureLogging(process, service=True)

def configureLoggingDiscovery(process):
    return configureLogging(process, discovery=True)

def configureLoggingAll(process):
    return configureLogging(process, client=True, server=True, service=True, discovery=True)
