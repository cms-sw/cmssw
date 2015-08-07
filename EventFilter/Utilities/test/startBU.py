import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

#Instructions:
#     before running this script make sure that you have access to /fff/BU0/ramdisk and /fff/data
#     these parameters can also be changed in startFU.py and startBU.py scripts
#user = os.environ['USER']

options = VarParsing.VarParsing ('analysis')

options.register ('runNumber',
                  100, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('buBaseDir',
                  '/fff/BU0/ramdisk/', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.parseArguments()

cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("FAKEBU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    multiProcesses = cms.untracked.PSet(
    maxChildProcesses = cms.untracked.int32(0)
    )
)

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring( 'cout' ),
                                    cout = cms.untracked.PSet( FwkReport =
                                                               cms.untracked.PSet(reportEvery = cms.untracked.int32(1000),
                                                                                  optionalPSet = cms.untracked.bool(True),
                                                                                  limit = cms.untracked.int32(10000000)
                                                                                  ),
                                                               threshold = cms.untracked.string( "INFO" )
                                                               )
                                    )

process.source = cms.Source("EmptySource",
     firstRun= cms.untracked.uint32(options.runNumber),
     numberEventsInLuminosityBlock = cms.untracked.uint32(200),
     numberEventsInRun       = cms.untracked.uint32(0)    
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.buBaseDir),
    buBaseDir = cms.untracked.string(options.buBaseDir),
    directorIsBu = cms.untracked.bool(True),
    copyRunDir = cms.untracked.bool(False))
    #obsolete:
    #smBaseDir  = cms.untracked.string("/fff/data/sm"),
    #slaveResources = cms.untracked.vstring('localhost'),
    #slavePathToData = cms.untracked.string("/fff/data/slave"))

process.EvFBuildingThrottle = cms.Service("EvFBuildingThrottle",
    highWaterMark = cms.untracked.double(0.80),
    lowWaterMark = cms.untracked.double(0.75),
    sleepmSecs  = cms.untracked.uint32(500))

process.a = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(0))

process.s = cms.EDProducer("DaqFakeReader",
                           meanSize = cms.untracked.uint32(1024),
                           width = cms.untracked.uint32(512),
                           injectErrPpm = cms.untracked.uint32(0)
                           )

process.out = cms.OutputModule("RawStreamFileWriterForBU",
    ProductLabel = cms.untracked.string("s"),
    numWriters = cms.untracked.uint32(1),
    eventBufferSize = cms.untracked.uint32(100),
    numEventsPerFile= cms.untracked.uint32(20),
    jsonDefLocation = cms.untracked.string(cmsswbase+"/src/EventFilter/Utilities/plugins/budef.jsd"),
    jsonEoLDefLocation = cms.untracked.string(cmsswbase+"/src/EventFilter/Utilities/plugins/eols.jsd"),
    frdVersion=cms.untracked.uint32(5),
    debug = cms.untracked.bool(True))

process.p = cms.Path(process.s+process.a)

process.ep = cms.EndPath(process.out)
