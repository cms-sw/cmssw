import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register( 'runNumber'
                , 1 #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Run number to be uploaded."
                  )
options.parseArguments()

process = cms.Process( "RunInfoPopulator" )
process.load( "CondCore.DBCommon.CondDBCommon_cfi" )
process.CondDBCommon.connect = 'sqlite_file:dbox_upload.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.DBParameters.messageLevel=cms.untracked.int32( 3 )

process.MessageLogger = cms.Service( "MessageLogger"
                                   , cout = cms.untracked.PSet( threshold = cms.untracked.string( 'INFO' ) )
                                   , destinations = cms.untracked.vstring( 'cout' )
                                     )

process.source = cms.Source( "EmptyIOVSource"
                           , lastValue = cms.uint64( 1 )
                           , timetype = cms.string( 'runnumber' )
                           , firstValue = cms.uint64( 1 )
                           , interval = cms.uint64( 1 )
                             )

process.PoolDBOutputService = cms.Service( "PoolDBOutputService"
                                         , process.CondDBCommon
                                         , logconnect = cms.untracked.string( 'sqlite_file:logruninfo_pop_test.db' )
                                         , timetype = cms.untracked.string( 'runnumber' )
                                         , toPut = cms.VPSet( cms.PSet( record = cms.string( 'RunInfoRcd' )
                                                                      , tag = cms.string( 'runinfo_31X_hlt' )
                                                                        )
                                                              )
                                          )

process.popConRunInfo = cms.EDAnalyzer( "RunInfoPopConAnalyzer"
                                      , SinceAppendMode = cms.bool( True )
                                      , record = cms.string( 'RunInfoRcd' )
                                      , Source = cms.PSet( runNumber = cms.uint64( options.runNumber )
                                                         , OnlineDBPass = cms.untracked.string( 'PASSWORD' )
                                                           )
                                      , loggingOn = cms.untracked.bool( True )
                                      , IsDestDbCheckedInQueryLog = cms.untracked.bool( False )
                                      , targetDBConnectionString = cms.untracked.string( 'sqlite_file:run_info_popcontest.db' )
                                        )

process.p = cms.Path( process.popConRunInfo )
