import time

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('runNumber',
                 100000, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('messageLevel',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Message level; default to 0")
options.register('globalTag',
                 'START70_V2::All', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.register('pfnPrefix',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "PFN prefix in GlobalTag connection strings")
options.register('pfnPostfix',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "PFN postfix in GlobalTag connection strings")
options.register('refresh',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Refresh type: default no refresh")
options.register('eventsPerLumi',
                 100, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events per lumi")
options.register('numberOfLumis',
                 100, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of lumisections per run")
options.register('numberOfRuns',
                 100, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of runs in the job")
options.parseArguments()

process = cms.Process("TEST")

process.MessageLogger = cms.Service( "MessageLogger",
                                     destinations = cms.untracked.vstring( 'detailedInfo' ),
                                     detailedInfo = cms.untracked.PSet( threshold = cms.untracked.string( 'INFO' ) ),
                                     )

process.add_( cms.Service( "PrintEventSetupDataRetrieval",
                           printProviders=cms.untracked.bool( True )
                           )
              )

CondDBSetup = cms.PSet( DBParameters = cms.PSet( authenticationPath = cms.untracked.string( '.' ),
                                                 connectionRetrialPeriod = cms.untracked.int32( 10 ),
                                                 idleConnectionCleanupPeriod = cms.untracked.int32( 10 ),
                                                 messageLevel = cms.untracked.int32( 0 ),
                                                 enablePoolAutomaticCleanUp = cms.untracked.bool( False ),
                                                 enableConnectionSharing = cms.untracked.bool( True ),
                                                 connectionRetrialTimeOut = cms.untracked.int32( 60 ),
                                                 connectionTimeOut = cms.untracked.int32( 0 ),
                                                 enableReadOnlySessionOnUpdateConnection = cms.untracked.bool( False )
                                                 )
                        )

CondDBSetup.DBParameters.messageLevel = options.messageLevel

refreshAlways, refreshOpenIOVs, refreshEachRun, reconnectEachRun = False, False, False, False
if options.refresh == 0:
    refreshAlways, refreshOpenIOVs, refreshEachRun, reconnectEachRun = False, False, False, False
elif options.refresh == 1:
    refreshAlways = True
    refreshOpenIOVs, refreshEachRun, reconnectEachRun = False, False, False
elif options.refresh == 2:
    refreshAlways = False
    refreshOpenIOVs = True
    refreshEachRun, reconnectEachRun = False, False
elif options.refresh == 3:
    refreshAlways, refreshOpenIOVs = False, False
    refreshEachRun = True
    reconnectEachRun = False
elif options.refresh == 4:
    refreshAlways, refreshOpenIOVs, refreshEachRun = False, False, False
    reconnectEachRun = True

process.GlobalTag = cms.ESSource( "PoolDBESSource",
                                  CondDBSetup,
                                  connect = cms.string( 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG' ),
                                  #connect = cms.string('sqlite_fip:CondCore/TagCollection/data/GlobalTag.db'), #For use during release integration
                                  globaltag = cms.string( 'UNSPECIFIED::All' ),
                                  RefreshAlways = cms.untracked.bool( refreshAlways ),
                                  RefreshOpenIOVs = cms.untracked.bool( refreshOpenIOVs ),
                                  RefreshEachRun=cms.untracked.bool( refreshEachRun ),
                                  ReconnectEachRun=cms.untracked.bool( reconnectEachRun ),
                                  DumpStat=cms.untracked.bool( True ),
                                  pfnPrefix=cms.untracked.string( '' ),   
                                  pfnPostfix=cms.untracked.string( '' )
                                  )

process.GlobalTag.globaltag = options.globalTag

if options.pfnPrefix:
    process.GlobalTag.pfnPrefix = options.pfnPrefix
if options.pfnPostfix:
    process.GlobalTag.pfnPostfix = options.pfnPostfix

#TODO: add VarParsing support for adding custom conditions
#process.GlobalTag.toGet = cms.VPSet()
#process.GlobalTag.toGet.append(
#   cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
#            tag = cms.string("firstcollisions"),
#             connect = cms.untracked.string("frontier://PromptProd/CMS_COND_31X_BEAMSPOT")
#           )
#)

process.source = cms.Source( "EmptySource",
                             firstRun = cms.untracked.uint32( options.runNumber ),
                             firstTime = cms.untracked.uint64( ( long( time.time() ) - 24 * 3600 ) << 32 ), #24 hours ago in nanoseconds
                             numberEventsInRun = cms.untracked.uint32( options.eventsPerLumi *  options.numberOfLumis ), # options.numberOfLumis lumi sections per run
                             numberEventsInLuminosityBlock = cms.untracked.uint32( options.eventsPerLumi )
                             )



process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( options.eventsPerLumi *  options.numberOfLumis * options.numberOfRuns ) ) #options.numberOfRuns runs per job

process.get = cms.EDAnalyzer( "EventSetupRecordDataGetter",
                              toGet =  cms.VPSet(),
                              verbose = cms.untracked.bool( True )
                             )

process.p = cms.Path( process.get )
