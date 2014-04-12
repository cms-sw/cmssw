import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" ),
                                                              FwkReport = cms.untracked.PSet(reportEvery = cms.untracked.int32(3000)),
                                                              ),
                                    destinations = cms.untracked.vstring( 'cout' ),
                                    categories = cms.untracked.vstring('FwkReport') 
                                    )


process.source = cms.Source("RawEventSourceFromBU",
                            fileNames = cms.untracked.vstring('file:pppp')
)
process.EvFDaqDirector = cms.Service("EvFDaqDirector",
                                     hltBaseDir = cms.untracked.string("/data/hlt"),
                                     smBaseDir  = cms.untracked.string("/data/sm")
                                     )
#process.EvFDaqDirector = cms.Service("EvFDaqDirector")
# process.EvFBuildingThrottle = cms.Service("EvFBuildingThrottle",
#                                           highWaterMark = cms.untracked.double(0.3),
#                                           lowWaterMark = cms.untracked.double(0.28)
#                                           )
process.a = cms.EDAnalyzer("ExceptionGenerator",
                           defaultAction = cms.untracked.int32(0),
                           defaultQualifier = cms.untracked.int32(10)
                           )
process.b = cms.EDAnalyzer("ExceptionGenerator",
                           defaultAction = cms.untracked.int32(0),
                           defaultQualifier = cms.untracked.int32(10)
                           )
process.c = cms.EDAnalyzer("ExceptionGenerator",
                           defaultAction = cms.untracked.int32(0),
                           defaultQualifier = cms.untracked.int32(10)
                           )
process.d = cms.EDAnalyzer("ExceptionGenerator",
                           defaultAction = cms.untracked.int32(0),
                           defaultQualifier = cms.untracked.int32(10)
                           )

process.psa = cms.EDFilter("HLTPrescaler",
                           L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                           offset = cms.uint32( 0 )
                           )
process.psb = cms.EDFilter("HLTPrescaler",
                           L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                           offset = cms.uint32( 1663 )
                           )
process.psc = cms.EDFilter("HLTPrescaler",
                           L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                           offset = cms.uint32( 1823 )
                           )
process.psd = cms.EDFilter("HLTPrescaler",
                           L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                           offset = cms.uint32( 1993 )
                           )

process.aa = cms.Path(process.psa*process.a)
process.bb = cms.Path(process.psb*process.b)
process.cc = cms.Path(process.psc*process.c)
process.dd = cms.Path(process.psd*process.d)

process.streamA = cms.OutputModule("Stream",
                                    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'aa' ) )
                                   )
process.streamB = cms.OutputModule("Stream",
                                    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'bb' ) )
                                   )
process.streamC = cms.OutputModule("Stream",
                                    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'cc' ) )
                                   )
process.streamD = cms.OutputModule("Stream",
                                    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'dd' ) )
                                   )

process.PrescaleService = cms.Service( "PrescaleService",
                                       forceDefault = cms.bool( False ),
                                       prescaleTable = cms.VPSet( 
    cms.PSet(  pathName = cms.string( "aa" ),
               prescales = cms.vuint32( 1, 0 )
               ),
    cms.PSet(  pathName = cms.string( "bb" ),
               prescales = cms.vuint32( 10, 0 )
               ),
    cms.PSet(  pathName = cms.string( "cc" ),
               prescales = cms.vuint32( 100, 0 )
               ),
    cms.PSet(  pathName = cms.string( "dd" ),
               prescales = cms.vuint32( 40, 40 )
               )
    ),
                                       lvl1DefaultLabel = cms.string( "3e33" ),
                                       lvl1Labels = cms.vstring( 'Cosmics',
                                                                 'Cosmics + High Random' )
                                       
                                       )

process.ep = cms.EndPath(process.streamA*process.streamB*process.streamC*process.streamD)
                                       
                                       
                                       
