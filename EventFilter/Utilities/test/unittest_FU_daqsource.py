from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

options = VarParsing.VarParsing ('analysis')

options.register ('runNumber',
                  100101, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('daqSourceMode',
                  '', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "DAQ source data mode")

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
                  2, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW threads")

options.register ('numFwkStreams',
                  2, # default value
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
    testing = cms.untracked.bool(True),
    dataMode = cms.untracked.string(options.daqSourceMode),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(False),
    eventChunkBlock = cms.untracked.uint32(2),
    eventChunkSize = cms.untracked.uint32(3),
    maxChunkSize = cms.untracked.uint32(10),
    numBuffers = cms.untracked.uint32(3),
    maxBufferedFiles = cms.untracked.uint32(2),
    fileListMode = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
        ram_dir_path + "run" + str(options.runNumber) + "_ls0001_index000000.raw",
        ram_dir_path + "run" + str(options.runNumber) + "_ls0001_index000001.raw",
        ram_dir_path + "run" + str(options.runNumber) + "_ls0002_index000000.raw",
        ram_dir_path + "run" + str(options.runNumber) + "_ls0002_index000001.raw"
    )

)

process.PrescaleService = cms.Service( "PrescaleService",
                                       forceDefault = cms.bool( False ),
                                       prescaleTable = cms.VPSet( 
                                         cms.PSet(  pathName = cms.string( "p1" ),
                                         prescales = cms.vuint32( 10)
                                         ),
                                         cms.PSet(  pathName = cms.string( "p2" ),
                                         prescales = cms.vuint32( 100 )
                                         )
                                       ),
                                       lvl1DefaultLabel = cms.string( "Default" ),
                                       lvl1Labels = cms.vstring( 'Default' )
                                       )

process.filter1 = cms.EDFilter("HLTPrescaler",
                               L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
                               )
process.filter2 = cms.EDFilter("HLTPrescaler",
                               L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
                               )

process.a = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(58))

process.b = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(5))

process.tcdsRawToDigi = cms.EDProducer("TcdsRawToDigi",
    InputLabel = cms.InputTag("rawDataCollector")
)

process.p1 = cms.Path(process.a*process.tcdsRawToDigi*process.filter1)
process.p2 = cms.Path(process.b*process.filter2)

process.streamA = cms.OutputModule("EvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p1' ))
)

process.streamB = cms.OutputModule("EvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p2' ))
)

process.streamC = cms.OutputModule("EvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p2' ))
)

process.streamD = cms.OutputModule("GlobalEvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p2' ))
)

process.hltJson = cms.EDAnalyzer("HLTriggerJSONMonitoring")

process.DQMStore = cms.Service( "DQMStore",
    verbose = cms.untracked.int32( 0 ),
    saveByLumi = cms.untracked.bool( False ),
)

from DQMServices.FileIO.DQMFileSaverPB_cfi import dqmSaver
process.hltDQMFileSaver = dqmSaver


process.daqHistoTest = cms.EDProducer("DaqTestHistograms",
    numberOfHistograms = cms.untracked.uint32(50),
    lumisectionRange =  cms.untracked.uint32(20)
)

process.ep = cms.EndPath(
  process.streamA
  + process.streamB
  + process.streamC
# + process.streamD
  + process.hltJson
  + process.daqHistoTest
  + process.hltDQMFileSaver
)
