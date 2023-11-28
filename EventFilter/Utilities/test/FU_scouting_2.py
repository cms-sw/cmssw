from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

options = VarParsing.VarParsing ('analysis')

options.register ('runNumber',
#                  325175, # default value
                  325172, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('buBaseDir',
                  'ramdisk', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fuBaseDir',
                  'data', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('fffBaseDir',
                  '.', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "FFF base directory")

options.register ('numThreads',
                  1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW threads")

options.register ('numFwkStreams',
                  1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW streams")


options.parseArguments()

cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("TESTFU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1) # ShmStreamConsumer requires synchronization at LuminosityBlock boundaries
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )),
    destinations = cms.untracked.vstring( 'cout' )
)

process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1)
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(False),
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

ram_dir_path=options.buBaseDir+"/run"+str(options.runNumber).zfill(6)+"/"

process.source = cms.Source("DAQSource",
#    dataMode = cms.untracked.string("ScoutingRun2Muon"),
    dataMode = cms.untracked.string("ScoutingRun2Multi"),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(False),
    eventChunkSize = cms.untracked.uint32(8),
    eventChunkBlock = cms.untracked.uint32(8),
    numBuffers = cms.untracked.uint32(2),
    maxBufferedFiles = cms.untracked.uint32(2),
    fileListMode = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
#        ram_dir_path+"run325175_ls0001_index000001.raw"
        ram_dir_path+"run325172_ls0010_index000000.raw",
        ram_dir_path+"run325172_ls0380_index000000.raw"
    )

)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolOutputTest.root'),
    outputCommands = cms.untracked.vstring("drop *","keep *_rawDataCollector_*_*")
    )


process.ep = cms.EndPath(
#  process.streamA
#  + process.streamB
#  + process.streamC
# + process.streamD
  process.output
)
