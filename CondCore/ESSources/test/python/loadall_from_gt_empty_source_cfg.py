import time

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('connectionString',
                 'frontier://FrontierProd/CMS_CONDITIONS', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag Connection string")
options.register('globalTag',
                 '80X_dataRun2_v4', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag")
options.register('snapshotTime',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "GlobalTag snapshot time")
options.register('refresh',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Refresh type: default no refresh")
options.register('pfnPostfix',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "PFN postfix in GlobalTag connection strings")
options.register('pfnPrefix',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "PFN prefix in GlobalTag connection strings")
options.register('runNumber',
                 4294967292, #default value, int limit -3
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('eventsPerLumi',
                 3, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events per lumi")
options.register('numberOfLumis',
                 3, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of lumisections per run")
options.register('numberOfRuns',
                 3, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of runs in the job")
options.register('messageLevel',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Message level; default to 0")
options.register('security',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "FroNTier connection security: activate it with 'sig'")

options.parseArguments()

process = cms.Process("TEST")

process.MessageLogger = cms.Service( "MessageLogger",
                                     destinations = cms.untracked.vstring( 'detailedInfo' ),
                                     detailedInfo = cms.untracked.PSet( threshold = cms.untracked.string( 'INFO' ) ),
                                     )

CondDBParameters = cms.PSet( authenticationPath = cms.untracked.string( '' ),
                             authenticationSystem = cms.untracked.int32( 0 ),
                             messageLevel = cms.untracked.int32( options.messageLevel ),
                             security = cms.untracked.string( options.security ),
                             )

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
                                  DBParameters = CondDBParameters,
                                  connect = cms.string( options.connectionString ),
                                  globaltag = cms.string( options.globalTag ),
                                  snapshotTime = cms.string( options.snapshotTime ),
                                  toGet = cms.VPSet(),
                                  RefreshAlways = cms.untracked.bool( refreshAlways ),
                                  RefreshOpenIOVs = cms.untracked.bool( refreshOpenIOVs ),
                                  RefreshEachRun = cms.untracked.bool( refreshEachRun ),
                                  ReconnectEachRun = cms.untracked.bool( reconnectEachRun ),
                                  DumpStat = cms.untracked.bool( True ),
                                  pfnPrefix = cms.untracked.string( '' ),   
                                  pfnPostfix = cms.untracked.string( '' )
                                  )

if options.pfnPrefix:
    process.GlobalTag.pfnPrefix = options.pfnPrefix
if options.pfnPostfix:
    process.GlobalTag.pfnPostfix = options.pfnPostfix

#TODO: add VarParsing support for adding custom conditions
#process.GlobalTag.toGet.append( cms.PSet( record = cms.string( "BeamSpotObjectsRcd" ),
#                                          tag = cms.string( "firstcollisions" ),
#                                          connect = cms.string( "frontier://FrontierProd/CMS_CONDITIONS" ),
#                                          snapshotTime = cms.string('2014-01-01 00:00:00.000'),
#                                          )
#                                )

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

process.escontent = cms.EDAnalyzer( "PrintEventSetupContent",
                                    compact = cms.untracked.bool( True ),
                                    printProviders = cms.untracked.bool( True )
                                    )

process.esretrieval = cms.EDAnalyzer( "PrintEventSetupDataRetrieval",
                                      printProviders = cms.untracked.bool( True )
                                      )

process.p = cms.Path( process.get )
process.esout = cms.EndPath( process.escontent + process.esretrieval )
if process.schedule_() is not None:
    process.schedule_().append( process.esout )

for name, module in process.es_sources_().iteritems():
    print "ESModules> provider:%s '%s'" % ( name, module.type_() )
for name, module in process.es_producers_().iteritems():
    print "ESModules> provider:%s '%s'" % ( name, module.type_() )
