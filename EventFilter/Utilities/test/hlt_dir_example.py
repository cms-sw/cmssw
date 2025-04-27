import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

options = VarParsing.VarParsing ('analysis')

#cmsRun runNumber=X ...
options.register ('runNumber',
                  18, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Run Number")

options.register ('daqSourceMode',
                  'DTH', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "DAQ source data mode")

options.register ('buBaseDir',
                  'ramdisk', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "BU base directory")

options.register ('fuBaseDir',
                  'data', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "BU base directory")

options.register ('fffBaseDir',
                  '.', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "FFF base directory")

options.register ('numThreads',
                  3,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of CMSSW threads")

options.register ('numFwkStreams',
                  2,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of CMSSW streams")

options.parseArguments()

process = cms.Process("DTHDEMO")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )),
    destinations = cms.untracked.vstring( 'cout' )
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    useFileBroker = cms.untracked.bool(True),
    fileBrokerHostFromCfg = cms.untracked.bool(False),
    fileBrokerHost = cms.untracked.string("htcp40.cern.ch"),
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fffBaseDir+"/"+options.fuBaseDir),
    buBaseDir = cms.untracked.string(options.fffBaseDir+"/"+options.buBaseDir),
    directorIsBU = cms.untracked.bool(False),

    #read data from single or multiple directories
    buBaseDirsAll = cms.untracked.vstring(options.fffBaseDir+"/"+options.buBaseDir),

    #number of sources read per each directory e.g. (2,4,1)
    buBaseDirsNumStreams = cms.untracked.vint32(1),

    #list of sources for each directory above in the same order (e.g. 1230,1231, 1255,1256,1257,1258, 1350)
    buBaseDirsStreamIDs = cms.untracked.vint32(1230),

    #naming convetion of the file (source, sourceid, fb, fbid, etc. or <empty> for non-DTH mode)
    sourceIdentifier = cms.untracked.string("source")

    #example for multiple sources in single directory:
    #buBaseDirsNumStreams = cms.untracked.vint32(3),
    #buBaseDirsStreamIDs = cms.untracked.vint32(1230),
    #buBaseDirsStreamIDs = cms.untracked.vint32(1232),
    #buBaseDirsStreamIDs = cms.untracked.vint32(1233),

)

process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1)
)

try:
  os.makedirs(options.fffBaseDir+"/"+options.fuBaseDir+"/run"+str(options.runNumber).zfill(6))
except Exception as ex:
  print(str(ex))
  pass

process.source = cms.Source("DAQSource",
    fileDiscoveryMode = cms.untracked.bool(True),
    fileListMode = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(),
    #testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(True if options.daqSourceMode != "DTH" else False),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(2),
    eventChunkSize = cms.untracked.uint32(3),
    maxChunkSize = cms.untracked.uint32(10),
    numBuffers = cms.untracked.uint32(3),
    maxBufferedFiles = cms.untracked.uint32(2),
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(f'file:DTH_dump{options.runNumber}.root'),
    outputCommands = cms.untracked.vstring(
        "keep *",
    ),
    compressionAlgorithm = cms.untracked.string("ZSTD"),
    compressionLevel = cms.untracked.int32(4),
)

process.ep = cms.EndPath(
  process.out
)
