import socket
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
process = cms.Process("BeamSpotOnlineStartRun")
from CondCore.CondDB.CondDB_cfi import *

options = VarParsing.VarParsing()
options.register( 'destinationConnection'
                , 'sqlite_file:beamspot_test.db' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads will be possibly written."
                  )
options.register( 'targetConnection'
                , '' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """Connection string to the target DB:
                     if not empty (default), this provides the latest IOV and payloads to compare;
                     it is the DB where payloads should be finally uploaded."""
                  )
options.register( 'tag'
                , 'beamspot_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag written in destinationConnection and finally appended in targetConnection."
                  )
options.register( 'sourceTag'
                , ''
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag used to retrieve the source payload from the targetConnection."
                  )
options.register( 'runNumber'
                , 1 #default value                                                                                                                                                          
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Run number for the target since."
                  )
options.register( 'messageLevel'
                , 0 #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Message level; default to 0"
                  )
options.parseArguments()

CondDBConnection = CondDB.clone( connect = cms.string( options.destinationConnection ) )
CondDBConnection.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBConnection,
                                          timetype = cms.untracked.string('lumiid'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('BeamSpotRcd'),
                                                                     tag = cms.string( options.tag )
                                                                     )
                                                            )
                                          )

process.PopCon = cms.EDAnalyzer("BeamSpotOnlinePopConAnalyzer",
                                SinceAppendMode = cms.bool(True),
                                record = cms.string('BeamSpotRcd'),
                                name = cms.untracked.string('BeamSpotOnline'),
                                Source = cms.PSet(
                                    runNumber = cms.untracked.uint32( options.runNumber ),
                                    sourcePayloadTag = cms.untracked.string( options.sourceTag ),
                                    # maxAge = one day in seconds
                                    maxAge = cms.untracked.uint32( 86400 ),
                                    debug=cms.untracked.bool(False)
                                ),
                               loggingOn = cms.untracked.bool(True),
                               IsDestDbCheckedInQueryLog = cms.untracked.bool(False)
                               )

process.p = cms.Path(process.PopCon)
