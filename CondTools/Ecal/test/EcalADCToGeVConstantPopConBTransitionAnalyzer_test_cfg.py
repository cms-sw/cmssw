import socket
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from CondCore.CondDB.CondDB_cfi import *

options = VarParsing.VarParsing()
options.register( 'runNumber'
                , 1 #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Run number to be uploaded."
                  )
options.register( 'destinationConnection'
                , 'sqlite_file:EcalADCToGeVConstant_PopCon_test.db' #default value
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
options.register( 'sourceConnection'
                , '' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , """Connection string to the DB hosting tags for RunInfo and EcalADCToGeVConstant.
                     It defaults to the same as the target connection, i.e. empty.
                     If target connection is also empty, it is set to be the same as destination connection."""
                  )
options.register( 'tag'
                , 'EcalADCToGeVConstant_PopCon_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag written in destinationConnection and finally appended in targetConnection."
                  )
options.register( 'tagForRunInfo'
                , 'runInfo_31X_hlt'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag used to retrieve the RunInfo payload and the magnet current therein."
                  )
options.register( 'tagForBOff'
                , 'EcalADCToGeVConstant_0T_test0'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag used to retrieve the EcalADCToGeVConstant payload for magnet off."
                  )
options.register( 'tagForBOn'
                , 'EcalADCToGeVConstant_3.8T_test0'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag used to retrieve the EcalADCToGeVConstant payload for magnet on."
                  )
options.register( 'currentThreshold'
                , 18000.
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.float
                , "The threshold on the magnet current for considering a switch of the magnetic field."
                  )
options.register( 'messageLevel'
                , 0 #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Message level; default to 0"
                  )
options.parseArguments()

if not options.sourceConnection:
    if options.targetConnection:
        options.sourceConnection = options.targetConnection
    else:
        options.sourceConnection = options.destinationConnection

CondDBConnection = CondDB.clone( connect = cms.string( options.destinationConnection ) )
CondDBConnection.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )

PopConConnection = CondDB.clone( connect = cms.string( options.sourceConnection ) )
PopConConnection.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )

process = cms.Process( "EcalADCToGeVConstantPopulator" )

process.MessageLogger = cms.Service( "MessageLogger"
                                   , destinations = cms.untracked.vstring( 'cout' )
                                   , cout = cms.untracked.PSet( threshold = cms.untracked.string( 'INFO' ) )
                                     )

if options.messageLevel == 3:
    #enable LogDebug output: remember the USER_CXXFLAGS="-DEDM_ML_DEBUG" compilation flag!
    process.MessageLogger.cout = cms.untracked.PSet( threshold = cms.untracked.string( 'DEBUG' ) )
    process.MessageLogger.debugModules = cms.untracked.vstring( '*' )

process.source = cms.Source( "EmptyIOVSource"
                           , lastValue = cms.uint64( options.runNumber )
                           , timetype = cms.string( 'runnumber' )
                           , firstValue = cms.uint64( options.runNumber )
                           , interval = cms.uint64( 1 )
                             )

process.PoolDBOutputService = cms.Service( "PoolDBOutputService"
                                         , CondDBConnection
                                         , timetype = cms.untracked.string( 'runnumber' )
                                         , toPut = cms.VPSet( cms.PSet( record = cms.string( 'EcalADCToGeVConstantRcd' )
                                                                      , tag = cms.string( options.tag )
                                                                        )
                                                              )
                                          )

process.popConEcalADCToGeVConstant = cms.EDAnalyzer( "EcalADCToGeVConstantPopConBTransitionAnalyzer"
                                                   , SinceAppendMode = cms.bool( True )
                                                   , record = cms.string( 'EcalADCToGeVConstantRcd' )
                                                   , Source = cms.PSet( BTransition = cms.PSet( PopConConnection
                                                                                              , runNumber = cms.uint64( options.runNumber )
                                                                                              , tagForRunInfo = cms.string( options.tagForRunInfo )
                                                                                              , tagForBOff = cms.string( options.tagForBOff )
                                                                                              , tagForBOn = cms.string( options.tagForBOn )
                                                                                              , currentThreshold = cms.untracked.double( options.currentThreshold )
                                                                                                )
                                                                        )
                                                   , loggingOn = cms.untracked.bool( True )
                                                   , targetDBConnectionString = cms.untracked.string( options.targetConnection )
                                                     )

process.p = cms.Path( process.popConEcalADCToGeVConstant )
