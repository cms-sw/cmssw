from __future__ import print_function
import time

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import six

options = VarParsing.VarParsing()
options.register('connectionString',
                 'frontier://FrontierProd/CMS_CONDITIONS', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Connection string')
options.register('record',
                 'EcalPedestalsRcd', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Event Setup Record registered for Condition usage')
options.register('label',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Label associated to the Record')
options.register('tag',
                 'EcalPedestals_205858_200_mc_reduced_noise', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Tag stored in Condition Database')
options.register('snapshotTime',
                 '9999-12-31 23:59:59.000', #default value is MAX_TIMESTAMP
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 'Tag snapshot time')
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
                                     destinations = cms.untracked.vstring( 'recordInfo' ),
                                     recordInfo = cms.untracked.PSet( threshold = cms.untracked.string( 'INFO' ) ),
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
                                  toGet = cms.VPSet( cms.PSet( record = cms.string( options.record ),
                                                               label = cms.untracked.string( options.label ),
                                                               connect = cms.string( options.connectionString ),
                                                               tag = cms.string( options.tag ),
                                                               snapshotTime = cms.string( options.snapshotTime ),
                                                               ),
                                                     ),
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

for name, module in six.iteritems(process.es_sources_()):
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
for name, module in six.iteritems(process.es_producers_()):
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
