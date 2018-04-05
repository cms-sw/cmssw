import socket
import time
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from CondCore.CondDB.CondDB_cfi import *
from Configuration.AlCa.autoCond import autoCond

options = VarParsing.VarParsing()
options.register('connectionString',
                 'frontier://FrontierProd/CMS_CONDITIONS', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag Connection string")
options.register('globalTag',
                 autoCond['run2_data'], #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.register( 'runNumber'
                , 1 #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.int
                , "Run number to be uploaded."
                  )
options.register( 'destinationConnection'
                , 'sqlite_file:EcalADCToGeVConstant_EDAnalyzer_test.db' #default value
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Connection string to the DB where payloads will be possibly written."
                  )
options.register( 'tag'
                , 'EcalADCToGeVConstant_EDAnalyzer_test'
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "Tag written in destinationConnection and finally appended onto the tag in connectionString."
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
                , "Message level; default to 0."
                  )
options.parseArguments()

CondDBConnection = CondDB.clone( connect = cms.string( options.connectionString ) )
CondDBConnection.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )
DestConnection = CondDB.clone( connect = cms.string( options.destinationConnection ) )
DestConnection.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )

process = cms.Process( "EcalADCToGeVConstantWriter" )

process.MessageLogger = cms.Service( "MessageLogger"
                                   , destinations = cms.untracked.vstring( 'cout' )
                                   , cout = cms.untracked.PSet( threshold = cms.untracked.string( 'INFO' ) )
                                     )

if options.messageLevel == 3:
    #enable LogDebug output: remember the USER_CXXFLAGS="-DEDM_ML_DEBUG" compilation flag!
    process.MessageLogger.cout = cms.untracked.PSet( threshold = cms.untracked.string( 'DEBUG' ) )
    process.MessageLogger.debugModules = cms.untracked.vstring( '*' )

process.source = cms.Source( "EmptySource",
                             firstRun = cms.untracked.uint32( options.runNumber ),
                             firstTime = cms.untracked.uint64( ( long( time.time() ) - 24 * 3600 ) << 32 ), #24 hours ago in nanoseconds
                             numberEventsInRun = cms.untracked.uint32( 1 ),
                             numberEventsInLuminosityBlock = cms.untracked.uint32( 1 )
                             )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1 ) )

process.GlobalTag = cms.ESSource( "PoolDBESSource",
                                  CondDBConnection,
                                  globaltag = cms.string( options.globalTag ),
                                  toGet = cms.VPSet()
                                  )
process.GlobalTag.toGet.append( cms.PSet( record = cms.string( "EcalADCToGeVConstantRcd" ),
                                          label = cms.untracked.string( "0T" ),
                                          tag = cms.string( "EcalADCToGeVConstant_0T_test0" ),
                                          connect = cms.string( "frontier://FrontierPrep/CMS_CONDITIONS" ),
                                          )
                                )
process.GlobalTag.toGet.append( cms.PSet( record = cms.string( "EcalADCToGeVConstantRcd" ),
                                          label = cms.untracked.string( "38T" ),
                                          tag = cms.string( "EcalADCToGeVConstant_3.8T_test0" ),
                                          connect = cms.string( "frontier://FrontierPrep/CMS_CONDITIONS" ),
                                          )
                                )

process.PoolDBOutputService = cms.Service( "PoolDBOutputService"
                                         , DestConnection
                                         , timetype = cms.untracked.string( 'runnumber' )
                                         , toPut = cms.VPSet( cms.PSet( record = cms.string( 'EcalADCToGeVConstantRcd' )
                                                                      , tag = cms.string( options.tag )
                                                                        )
                                                              )
                                          )

process.ecalADCToGeVConstantBTransition = cms.EDAnalyzer( "EcalADCToGeVConstantBTransitionAnalyzer"
                                                        , currentThreshold = cms.untracked.double( options.currentThreshold )
                                                          )

process.p = cms.Path( process.ecalADCToGeVConstantBTransition )
