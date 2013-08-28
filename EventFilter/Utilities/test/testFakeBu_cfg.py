import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    multiProcesses = cms.untracked.PSet(
    maxChildProcesses = cms.untracked.int32(1)
    )
)
process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string( "INFO" )
                                                              ),
                                    destinations = cms.untracked.vstring( 'cout' )
                                    )


process.source = cms.Source("DaqSource",
                            readerPluginName = cms.untracked.string('DaqFakeReader'),
                            readerPset = cms.untracked.PSet(),
                            secondsPerLS = cms.untracked.uint32(23),
                            runNumber = cms.untracked.uint32(200781)
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
                                     hltBaseDir = cms.untracked.string("/data/hlt"),
                                     smBaseDir  = cms.untracked.string("/data/sm"),
                                     directorIsBu = cms.untracked.bool(True)
                                     )
process.EvFBuildingThrottle = cms.Service("EvFBuildingThrottle",
                                          highWaterMark = cms.untracked.double(0.50),
                                          lowWaterMark = cms.untracked.double(0.45)
                                          )
process.a = cms.EDAnalyzer("ExceptionGenerator",
                           defaultAction = cms.untracked.int32(0),
                           defaultQualifier = cms.untracked.int32(10)
                           )


process.out = cms.OutputModule("RawStreamFileWriterForBU",
                               ProductLabel = cms.untracked.string("rawDataCollector")
                               )

#process.p = cms.Path(process.a)

process.ep = cms.EndPath(process.out)


