import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

options = VarParsing.VarParsing ('analysis')

options.register ('runNumber',
                  100, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.register ('buBaseDir',
                  '/fff/BU0/ramdisk', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")


options.register ('fuBaseDir',
                  '/fff/data', # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "BU base directory")

options.register ('numThreads',
                  1, # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Number of CMSSW threads")

options.parseArguments()

cmsswbase = os.path.expandvars("$CMSSW_BASE/")

process = cms.Process("TESTFU")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numThreads),
    numberOfStreams = cms.untracked.uint32(options.numThreads),
    multiProcesses = cms.untracked.PSet(
    maxChildProcesses = cms.untracked.int32(0)
    )
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )),
    destinations = cms.untracked.vstring( 'cout' ))

process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1),
    microstateDefPath = cms.untracked.string( cmsswbase+'/src/EventFilter/Utilities/plugins/microstatedef.jsd'),
    fastMicrostateDefPath = cms.untracked.string( cmsswbase+'/src/EventFilter/Utilities/plugins/microstatedeffast.jsd'),
    fastName = cms.untracked.string( 'fastmoni' ),
    slowName = cms.untracked.string( 'slowmoni' ))

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    runNumber = cms.untracked.uint32(options.runNumber),
    baseDir = cms.untracked.string(options.fuBaseDir),
    buBaseDir = cms.untracked.string(options.buBaseDir),
    directorIsBu = cms.untracked.bool(False),
    testModeNoBuilderUnit = cms.untracked.bool(False))

try:
  os.makedirs(options.fuBaseDir+"/run"+str(options.runNumber).zfill(6))
except Exception, ex:
  print str(ex)
  pass

process.PrescaleService = cms.Service( "PrescaleService",
    lvl1DefaultLabel = cms.string( "B" ),
    lvl1Labels = cms.vstring( 'A',
                              'B'
                            ),
    prescaleTable = cms.VPSet(
               cms.PSet(  pathName = cms.string( "p1" ),                                                                                                                
                          prescales = cms.vuint32( 0, 10)
                       ),                                                                                                                                   
               cms.PSet(  pathName = cms.string( "p2" ),                                                                                                           
                          prescales = cms.vuint32( 0, 100)                                                                                                                   
                       )
    ))

process.source = cms.Source("FedRawDataInputSource",
    runNumber = cms.untracked.uint32(options.runNumber),
    getLSFromFilename = cms.untracked.bool(True),
    testModeNoBuilderUnit = cms.untracked.bool(False),
    verifyAdler32 = cms.untracked.bool(True),
    verifyChecksum = cms.untracked.bool(True),
    useL1EventID = cms.untracked.bool(True),
    eventChunkSize = cms.untracked.uint32(16),
    numBuffers = cms.untracked.uint32(2),
    eventChunkBlock = cms.untracked.uint32(1)
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

process.p1 = cms.Path(process.a*process.filter1)
process.p2 = cms.Path(process.b*process.filter2)

process.streamA = cms.OutputModule("EvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p1' ))
)

process.streamB = cms.OutputModule("EvFOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p2' ))
)

process.ep = cms.EndPath(process.streamA+process.streamB)
