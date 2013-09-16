import FWCore.ParameterSet.Config as cms

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
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )),
    destinations = cms.untracked.vstring( 'cout' ))


process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('DaqFakeReader'),
    readerPset = cms.untracked.PSet(),
    secondsPerLS = cms.untracked.uint32(5),
    runNumber = cms.untracked.uint32(100)
)


process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    baseDir = cms.untracked.string("ram"),
    buBaseDir = cms.untracked.string("ram"),
    smBaseDir  = cms.untracked.string("hdd"),
    directorIsBu = cms.untracked.bool(True),
    copyRunDir = cms.untracked.bool(True),	
    slaveResources = cms.untracked.vstring('localhost'),
    slavePathToData = cms.untracked.string("/home/aspataru/cmssw/CMSSW_6_2_0_pre3/src/andrei_configs/singleMachineTest/FU/hdd"))

process.EvFBuildingThrottle = cms.Service("EvFBuildingThrottle",
    highWaterMark = cms.untracked.double(0.50),
    lowWaterMark = cms.untracked.double(0.45))

process.a = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(5))

process.out = cms.OutputModule("RawStreamFileWriterForBU",
    ProductLabel = cms.untracked.string("rawDataCollector"),
    numWriters = cms.untracked.uint32(1),
    eventBufferSize = cms.untracked.uint32(100),
    numEventsPerFile= cms.untracked.uint32(20),
    jsonDefLocation = cms.untracked.string("/home/aspataru/cmssw/CMSSW_6_2_0_pre3/src/EventFilter/Utilities/plugins/budef.jsd"),
    #lumiSubdirectoriesMode=cms.untracked.bool(False),
    debug = cms.untracked.bool(True))

process.p = cms.Path(process.a)

process.ep = cms.EndPath(process.out)
