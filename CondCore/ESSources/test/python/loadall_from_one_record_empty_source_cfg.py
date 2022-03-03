
from __future__ import print_function
import time

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.AlCa.autoCond import autoCond

options = VarParsing.VarParsing()
options.register('processId',
                 '0',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Process Id")
options.register('connectionString',
                 'frontier://FrontierPrep/CMS_CONDITIONS',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "CondDB Connection string")
options.register('tag',
                 'BeamSpot_test_updateByLumi_00',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "tag for record BeamSpotObjectsRcd")
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
options.register('runNumber',
                 120013, #default value, int limit -3
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('eventsPerLumi',
                 20, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events per lumi")
options.register('numberOfLumis',
                 20, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of lumisections per run")
options.register('numberOfRuns',
                 1, #default value
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

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )

CondDBParameters = cms.PSet( authenticationPath = cms.untracked.string( '' ),
                             authenticationSystem = cms.untracked.int32( 0 ),
                             messageLevel = cms.untracked.int32( 3 ),
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
                                  snapshotTime = cms.string( options.snapshotTime ),
                                  frontierKey = cms.untracked.string('abcdefghijklmnopqrstuvwxyz0123456789'),
                                  toGet = cms.VPSet(cms.PSet(
                                      record = cms.string('BeamSpotObjectsRcd'),
                                      tag = cms.string( options.tag ),
                                      refreshTime = cms.uint64( 2 )
                                  )),
                                  RefreshAlways = cms.untracked.bool( refreshAlways ),
                                  RefreshOpenIOVs = cms.untracked.bool( refreshOpenIOVs ),
                                  RefreshEachRun = cms.untracked.bool( refreshEachRun ),
                                  ReconnectEachRun = cms.untracked.bool( reconnectEachRun ),
                                  DumpStat = cms.untracked.bool( True ),
                                  )

process.source = cms.Source( "EmptySource",
                             firstRun = cms.untracked.uint32( options.runNumber ),
                             firstLuminosityBlock = cms.untracked.uint32( 1 ),
                             numberEventsInRun = cms.untracked.uint32( options.eventsPerLumi *  options.numberOfLumis ), # options.numberOfLumis lumi sections per run
                             numberEventsInLuminosityBlock = cms.untracked.uint32( options.eventsPerLumi )
                             )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( options.eventsPerLumi *  options.numberOfLumis * options.numberOfRuns ) ) #options.numberOfRuns runs per job

process.prod = cms.EDAnalyzer("LumiTestReadAnalyzer",
                              processId = cms.untracked.string( options.processId ),
                              pathForLastLumiFile = cms.untracked.string("last_lumi.txt"),
                              pathForAllLumiFile = cms.untracked.string("./all_time.txt" ),
                              pathForErrorFile = cms.untracked.string("./lumi_read_errors")
)

process.p = cms.Path( process.prod )

for name, module in process.es_sources_().items():
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
for name, module in process.es_producers_().items():
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
