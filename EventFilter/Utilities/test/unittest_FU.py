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
                  3, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW threads")

options.register ('numFwkStreams',
                  2, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW streams")

options.register ('numEventsToWrite',
                 -1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                 "Number of Events to process. -1 means all.")


options.parseArguments()

cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("TESTFU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.numEventsToWrite)
)
if options.numEventsToWrite == 0:
  process.maxEvents.input = 1

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numFwkStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(2)
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "ERROR" )),
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

process.source = cms.Source("FedRawDataInputSource",
    getLSFromFilename = cms.untracked.bool(True),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(False),
    eventChunkSize = cms.untracked.uint32(8),
    eventChunkBlock = cms.untracked.uint32(8),
    numBuffers = cms.untracked.uint32(2),
    maxBufferedFiles = cms.untracked.uint32(2),
    fileListMode = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
        ram_dir_path+"run100101_ls0001_index000000.raw",
        ram_dir_path+"run100101_ls0001_index000001.raw",
        ram_dir_path+"run100101_ls0002_index000000.raw",
        ram_dir_path+"run100101_ls0002_index000001.raw"
    )

)

process.PrescaleService = cms.Service( "PrescaleService",
                                       forceDefault = cms.bool( False ),
                                       prescaleTable = cms.VPSet( 
                                         cms.PSet(  pathName = cms.string( "HLT_Physics" ),
                                         prescales = cms.vuint32( 10)
                                         ),
                                         cms.PSet(  pathName = cms.string( "HLT_Muon" ),
                                         prescales = cms.vuint32( 100 )
                                         )
                                       ),
                                       lvl1DefaultLabel = cms.string( "Default" ),
                                       lvl1Labels = cms.vstring( 'Default' )
                                       )

#used in case where we write no events
process.pre = cms.EDFilter("PrescaleEventFilter", offset = cms.uint32(0), prescale=cms.uint32(1))
if options.numEventsToWrite:
  process.pre.offset = 2
  process.pre.prescale = 4

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


import EventFilter.OnlineMetaDataRawToDigi.tcdsRawToDigi_cfi
process.tcdsRawToDigi = EventFilter.OnlineMetaDataRawToDigi.tcdsRawToDigi_cfi.tcdsRawToDigi.clone()
process.tcdsRawToDigi.InputLabel = cms.InputTag("rawDataCollector")

process.HLT_Physics = cms.Path(process.a*process.tcdsRawToDigi*process.filter1*process.pre)
process.HLT_Muon = cms.Path(process.b*process.filter2*process.pre)

process.streamA = cms.OutputModule("GlobalEvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'HLT_Physics' ))
)

process.streamB = cms.OutputModule("GlobalEvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'HLT_Muon' ))
)

process.streamDQM = cms.OutputModule("GlobalEvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'HLT_Physics', 'HLT_Muon' ))
)

process.streamD = cms.OutputModule("EventStreamFileWriter",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'HLT_Muon' ))
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
  + process.streamDQM
# + process.streamD
  + process.hltJson
  + process.daqHistoTest
  + process.hltDQMFileSaver
)
