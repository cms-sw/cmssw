
from __future__ import print_function
import time

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.AlCa.autoCond import autoCond

options = VarParsing.VarParsing()
options.register('connectionString',
                 'sqlite:cms_conditions.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "CondDB Connection string")
options.register('tag',
                 'BeamSpot_test_updateByLumi_00',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "tag for record BeamSpotObjectsRcd")
options.register('runNumber',
                 250000, #default value, int limit -3
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

options.parseArguments()

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )

CondDBParameters = cms.PSet( authenticationPath = cms.untracked.string( '/build/gg/' ),
                             authenticationSystem = cms.untracked.int32( 0 ),
                             messageLevel = cms.untracked.int32( 1 ),
                             )


process.GlobalTag = cms.ESSource( "PoolDBESSource",
                                  DBParameters = CondDBParameters,
                                  connect = cms.string( options.connectionString ),
                                  frontierKey = cms.untracked.string('abcdefghijklmnopqrstuvwxyz0123456789'),
                                  toGet = cms.VPSet(cms.PSet(
                                      record = cms.string('BeamSpotObjectsRcd'),
                                      tag = cms.string( options.tag ),
                                      refreshTime = cms.uint64( 2 )
                                  )),
                                  DumpStat = cms.untracked.bool( True ),
                                  )


process.source = cms.Source( "EmptySource",
                             firstRun = cms.untracked.uint32( options.runNumber ),
                             firstLuminosityBlock = cms.untracked.uint32( 1 ),
                             #firstTime = cms.untracked.uint64( 5401426372679696384 ), 
                             #firstTime = cms.untracked.uint64( 5771327162577584128 ), 
                             #timeBetweenEvents = cms.untracked.uint64( 429496729600 ),
                             numberEventsInRun = cms.untracked.uint32( 240 ), # options.numberOfLumis lumi sections per run
#                             numberEventsInRun = cms.untracked.uint32( options.eventsPerLumi *  options.numberOfLumis ), # options.numberOfLumis lumi sections per run
                             numberEventsInLuminosityBlock = cms.untracked.uint32( 10 )
                             )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))

process.prod = cms.EDAnalyzer("LumiTestWriteReadAnalyzer",
)

process.p = cms.Path( process.prod )

for name, module in process.es_sources_().items():
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
for name, module in process.es_producers_().items():
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
