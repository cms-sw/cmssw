from FWCore.ParameterSet.VarParsing import VarParsing
import FWCore.ParameterSet.Config as cms
import os, sys, json

options = VarParsing("analysis")
options.register("address", "0.0.0.0", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("port", 8001, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("timeout", 30, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("params", "", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("threads", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("streams", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("batchSize", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int)
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

# check mode
allowed_modes = {
    "Async": "TritonImageProducerAsync",
    "Sync": "TritonImageProducerSync",
    "PseudoAsync": "TritonImageProducerPseudoAsync",
}
if options.mode not in allowed_modes:
    raise ValueError("Unknown mode: "+options.mode)

process = cms.Process('imageTest')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("EmptySource")

process.TritonImageProducer = cms.EDProducer(allowed_modes[options.mode],
    topN = cms.uint32(5),
    imageList = cms.string("../data/models/resnet50_netdef/resnet50_labels.txt"),
    Client = cms.PSet(
        batchSize = cms.untracked.uint32(options.batchSize),
        address = cms.untracked.string(options.address),
        port = cms.untracked.uint32(options.port),
        timeout = cms.untracked.uint32(options.timeout),
        modelName = cms.string("resnet50_netdef"),
        modelVersion = cms.int32(-1),
        verbose = cms.untracked.bool(options.verbose),
        allowedTries = cms.untracked.uint32(0),
    )
)

# Let it run
process.p = cms.Path(
    process.TritonImageProducer
)

process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 500
keep_msgs = ['TritonImageProducer','TritonImageProducer:TritonClient','TritonClient']
for msg in keep_msgs:
    process.MessageLogger.categories.append(msg)
    setattr(process.MessageLogger.cerr,msg,
        cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000),
        )
    )

if options.threads>0:
    process.options.numberOfThreads = options.threads
    process.options.numberOfStreams = options.streams

