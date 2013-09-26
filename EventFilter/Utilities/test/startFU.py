import FWCore.ParameterSet.Config as cms

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
    microstateDefPath = cms.untracked.string( '/home/aspataru/cmssw/CMSSW_6_2_0_pre3/src/EventFilter/Utilities/plugins/microstatedef.jsd' ),
    outputDefPath = cms.untracked.string( '/home/aspataru/cmssw/CMSSW_6_2_0_pre3/src/EventFilter/Utilities/plugins/output.jsd' ),
    fastName = cms.untracked.string( 'fastmoni' ),
    slowName = cms.untracked.string( 'slowmoni' ))

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    baseDir = cms.untracked.string("hdd"),
    buBaseDir = cms.untracked.string("/home/aspataru/cmssw/CMSSW_6_2_0_pre3/src/andrei_configs/singleMachineTest/BU/ram"),
    smBaseDir  = cms.untracked.string("hdd"),
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
    rootBUDirectory = cms.untracked.string("/home/aspataru/cmssw/CMSSW_6_2_0_pre3/src/andrei_configs/singleMachineTest/BU/ram"),
    getLSFromFilename = cms.untracked.bool(True),
    testModeNoBuilderUnit = cms.untracked.bool(False),
    eventChunkSize = cms.untracked.uint32(16))


process.filter1 = cms.EDFilter("HLTPrescaler",
                               prescaleFactor = cms.int32(-1),
                               L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" )
                               )
process.filter2 = cms.EDFilter("HLTPrescaler",
                               prescaleFactor = cms.int32(-1),
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
