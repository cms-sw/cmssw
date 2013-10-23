import FWCore.ParameterSet.Config as cms
import os

output_def_location = os.path.expandvars("$CMSSW_BASE/src/EventFilter/Utilities/plugins/output.jsd")
microstate_def_location = os.path.expandvars("$CMSSW_BASE/src/EventFilter/Utilities/plugins/microstatedef.jsd")

process = cms.Process("TESTFU")
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

process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1),
    microstateDefPath = cms.untracked.string(microstate_def_location),
    outputDefPath = cms.untracked.string(output_def_location),
    fastName = cms.untracked.string( 'fastmoni' ),
    slowName = cms.untracked.string( 'slowmoni' ))

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    runNumber = cms.untracked.uint32(100),
    baseDir = cms.untracked.string("/data/hdd"),
    buBaseDir = cms.untracked.string("/dev/shm"),
    smBaseDir  = cms.untracked.string("/data/pippo"),
    directorIsBu = cms.untracked.bool(False),
    testModeNoBuilderUnit = cms.untracked.bool(False))

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
    runNumber = cms.untracked.uint32(100),
    getLSFromFilename = cms.untracked.bool(True),
    testModeNoBuilderUnit = cms.untracked.bool(False),
    eventChunkSize = cms.untracked.uint32(16))

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
    defaultQualifier = cms.untracked.int32(5))

process.b = cms.EDAnalyzer("ExceptionGenerator",
    defaultAction = cms.untracked.int32(0),
    defaultQualifier = cms.untracked.int32(5))


process.p1 = cms.Path(process.a*process.filter1)
process.p2 = cms.Path(process.b*process.filter2)


process.streamA = cms.OutputModule("Stream",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p1' ))
)

process.streamB = cms.OutputModule("Stream",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring( 'p2' ))
)
                                   

process.ep = cms.EndPath(process.streamA+process.streamB)
