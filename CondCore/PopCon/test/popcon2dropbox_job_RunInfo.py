import FWCore.ParameterSet.VarParsing as VarParsing
from CondCore.Utilities.popcon2dropbox_job_conf import md, process

options = VarParsing.VarParsing()
options.register( 'runNumber'
                       , 1 #default value
                       , VarParsing.VarParsing.multiplicity.singleton
                       , VarParsing.VarParsing.varType.int
                       , "Run number to be uploaded.")
options.parseArguments()

process.RunInfo = cms.EDAnalyzer( "RunInfoPopConAnalyzer"
                                                     , SinceAppendMode = cms.bool( True )
                                                     , record = cms.string( 'RunInfoRcd' )
                                                     , Source = cms.PSet( runNumber = cms.uint64( options.runNumber )
                                                                                    , OnlineDBPass = cms.untracked.string( 'MICKEY2MOUSE' ) )
                                                     , loggingOn = cms.untracked.bool( True )
                                                     , targetDBConnectionString = cms.untracked.string( str( md.destinationDatabase() ) )
                                                     , IsDestDbCheckedInQueryLog = cms.untracked.bool(False) )

process.p = cms.Path(process.RunInfo)
