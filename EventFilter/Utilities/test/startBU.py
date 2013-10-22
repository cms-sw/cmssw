import FWCore.ParameterSet.Config as cms
import os

budeflocation = os.path.expandvars("$CMSSW_BASE/src/EventFilter/Utilities/plugins/budef.jsd")

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
     firstRun= cms.untracked.uint32(100),
     numberEventsInLuminosityBlock = cms.untracked.uint32(200),
     numberEventsInRun       = cms.untracked.uint32(0)    
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    runNumber = cms.untracked.uint32(100),
    baseDir = cms.untracked.string("/dev/shm"),
    buBaseDir = cms.untracked.string("/dev/shm"),
    smBaseDir  = cms.untracked.string("/data/hdd"),
    directorIsBu = cms.untracked.bool(True),
    copyRunDir = cms.untracked.bool(False),	
    slaveResources = cms.untracked.vstring('localhost'),
    slavePathToData = cms.untracked.string("/data/hdd"))

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
    numEventsPerFile= cms.untracked.uint32(50),
    jsonDefLocation = cms.untracked.string(budeflocation),
    #lumiSubdirectoriesMode=cms.untracked.bool(False),
    debug = cms.untracked.bool(True))

process.p = cms.Path(process.s+process.a)

process.ep = cms.EndPath(process.out)
