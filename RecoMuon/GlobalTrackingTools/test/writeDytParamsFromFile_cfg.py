import FWCore.ParameterSet.Config as cms

process = cms.Process("WriteDYTParamsFromFile")

process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1),
                            firstRun = cms.untracked.uint32(1)
                            )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = "PRE_LS172_V15::All"

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:dytParamsFromFile.db'),
                                          authenticationMethod = cms.untracked.uint32(0),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('DYTParamsObjectRcd'),
                                                                     tag = cms.string('test')
                                                                     )
                                                            )
                                          )

process.dytParamsWriter = cms.EDAnalyzer("DYTParamsWriter",
                                        inputFileName = cms.string("dyt_params_example.txt"),
                                        inputFunction = cms.string("[0]x+[1]")
                                        )


process.clientPath = cms.Path(process.dytParamsWriter)
