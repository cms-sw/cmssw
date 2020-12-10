from FWCore.ParameterSet.VarParsing import VarParsing
import FWCore.ParameterSet.Config as cms
import os, sys, json

options = VarParsing("analysis")
options.register("serverName", "default", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("address", "0.0.0.0", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("port", 8001, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("timeout", 30, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("params", "", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("threads", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("streams", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("producer", "TritonImageProducer", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("modelName","resnet50_netdef", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("mode","PseudoAsync", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("verbose", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.parseArguments()

if len(options.params)>0:
    with open(options.params,'r') as pfile:
        pdict = json.load(pfile)
    options.address = pdict["address"]
    options.port = int(pdict["port"])
    print("server = "+options.address+":"+str(options.port))

# check producer/model
models = {
  "TritonImageProducer": "resnet50_netdef",
  "TritonGraphProducer": "gat_test",
}

if options.producer not in models:
    raise ValueError("Unknown producer: "+options.producer)

process = cms.Process('tritonTest')

process.load("HeterogeneousCore.SonicTriton.TritonService_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("EmptySource")

process.TritonService.verbose = options.verbose
process.TritonService.fallback.enable = True
process.TritonService.fallback.verbose = options.verbose
if len(options.address)>0:
    process.TritonService.servers.append(
        cms.PSet(
            name = cms.untracked.string(options.serverName),
            address = cms.untracked.string(options.address),
            port = cms.untracked.uint32(options.port),
        )
    )

process.TritonProducer = cms.EDProducer(options.producer,
    Client = cms.PSet(
        mode = cms.string(options.mode),
        preferredServer = cms.untracked.string(""),
        timeout = cms.untracked.uint32(options.timeout),
        modelName = cms.string(models[options.producer]),
        modelVersion = cms.string(""),
        modelConfigPath = cms.FileInPath("HeterogeneousCore/SonicTriton/data/models/{}/config.pbtxt".format(models[options.producer])),
        verbose = cms.untracked.bool(options.verbose),
        allowedTries = cms.untracked.uint32(0),
    )
)
if options.producer=="TritonImageProducer":
    process.TritonProducer.batchSize = cms.uint32(1)
    process.TritonProducer.topN = cms.uint32(5)
    process.TritonProducer.imageList = cms.string("../data/models/resnet50_netdef/resnet50_labels.txt")

# Let it run
process.p = cms.Path(
    process.TritonProducer
)

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 500
keep_msgs = [options.producer,options.producer+':TritonClient','TritonClient','TritonService']
for msg in keep_msgs:
    
    setattr(process.MessageLogger.cerr,msg,
        cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
        )
    )

if options.threads>0:
    process.options.numberOfThreads = options.threads
    process.options.numberOfStreams = options.streams

