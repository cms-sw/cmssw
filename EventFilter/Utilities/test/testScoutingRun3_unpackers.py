import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os, sys

options = VarParsing.VarParsing ("analysis")

options.register ("runNumber",
                  368636,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Run Number")

options.register ("daqSourceMode",
                  "ScoutingRun3",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "DAQ source data mode")

options.register ("buBaseDir",
                  "ramdisk",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "BU base directory")

options.register ("fuBaseDir",
                  "data",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "BU base directory")

options.register ("fffBaseDir",
                  "/tmp",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "FFF base directory")

options.register ("numThreads",
                  8,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of CMSSW threads")

options.register ("numFwkStreams",
                  8,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of CMSSW streams")

options.parseArguments()

cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("SCPU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string( "WARNING" )
    ),
    destinations = cms.untracked.vstring( "cout" ),
)

process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1)
)

process.Timing = cms.Service("Timing",
  summaryOnly = cms.untracked.bool(True),
  useJobReport = cms.untracked.bool(True)
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(False),
    buBaseDirsAll = cms.untracked.vstring(
        options.fffBaseDir+"/"+options.buBaseDir
    ),
    buBaseDirsNumStreams = cms.untracked.vint32(
        2
    ),
    fileBrokerHostFromCfg = cms.untracked.bool(True),
    fileBrokerHost = cms.untracked.string("htcp40.cern.ch"),
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fffBaseDir+"/"+options.fuBaseDir),
    buBaseDir = cms.untracked.string(options.fffBaseDir+"/"+options.buBaseDir),
    directorIsBU = cms.untracked.bool(False),
)

try:
  os.makedirs(options.fffBaseDir+"/"+options.fuBaseDir+"/run"+str(options.runNumber).zfill(6))
except Exception as ex:
  print(str(ex))
  pass

buDir = options.fffBaseDir+"/"+options.buBaseDir+("/run%06d" % options.runNumber)
if not os.path.isdir(buDir):
    os.makedirs(buDir)
os.system("touch " + buDir + "/" + "fu.lock")

flist = [
   buDir+"/" + "run" + str(options.runNumber) + "_ls0340_index000028.raw"
]

process.source = cms.Source("DAQSource",
    testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(False),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(64),
    eventChunkSize = cms.untracked.uint32(128),
    maxChunkSize = cms.untracked.uint32(256),
    numBuffers = cms.untracked.uint32(2),
    maxBufferedFiles = cms.untracked.uint32(2),
    fileListMode = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(*flist)
)

import EventFilter.L1ScoutingRawToDigi.ScGMTRawToDigi_cfi
process.GmtUnpacker = EventFilter.L1ScoutingRawToDigi.ScGMTRawToDigi_cfi.ScGmtUnpacker.clone()

import EventFilter.L1ScoutingRawToDigi.ScCaloRawToDigi_cfi
process.CaloUnpacker = EventFilter.L1ScoutingRawToDigi.ScCaloRawToDigi_cfi.ScCaloUnpacker.clone()

rawToDigiTask = cms.Task(
  process.GmtUnpacker,process.CaloUnpacker
)

process.p = cms.Path(rawToDigiTask)

process.outputZB = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:' + options.fffBaseDir + '/PoolOutputTest.root'),
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep *_GmtUnpacker_*_*",
        "keep *_CaloUnpacker_*_*"
    ),
    #compressionAlgorithm = cms.untracked.string("ZSTD"),
    #compressionLevel = cms.untracked.int32(4)
)

process.ep = cms.EndPath(
    process.outputZB
)
