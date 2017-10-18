import time
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from CondCore.CondDB.CondDB_cfi import *

options = VarParsing.VarParsing()
options.register('connectionString',
                 'sqlite_file:GEMEMap.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string")
options.register('tag',
                 'GEMEMap_v2', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Tag")
options.register('runNumber',
                 4294967292, #default value, int limit -3
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('eventsPerLumi',
                 1, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events per lumi")
options.register('numberOfLumis',
                 1, #default value
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

CondDBReference = CondDB.clone( connect = cms.string( options.connectionString ) )
CondDBReference.DBParameters.messageLevel = cms.untracked.int32( options.messageLevel )

process = cms.Process( "DBTest" )

process.MessageLogger = cms.Service( "MessageLogger",
                                     destinations = cms.untracked.vstring( 'cout' ),
                                     cout = cms.untracked.PSet( threshold = cms.untracked.string( 'INFO' ) ),
                                     )

process.source = cms.Source( "EmptySource",
                             firstRun = cms.untracked.uint32( options.runNumber ),
                             firstTime = cms.untracked.uint64( ( long( time.time() ) - 24 * 3600 ) << 32 ), #24 hours ago in nanoseconds
                             numberEventsInRun = cms.untracked.uint32( options.eventsPerLumi *  options.numberOfLumis ), # options.numberOfLumis lumi sections per run
                             numberEventsInLuminosityBlock = cms.untracked.uint32( options.eventsPerLumi )
                             )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( options.eventsPerLumi * options.numberOfLumis * options.numberOfRuns ) ) #options.numberOfRuns runs per job

process.GEMCabling = cms.ESSource( "PoolDBESSource",
                                   CondDBReference,
                                   toGet = cms.VPSet( cms.PSet( record = cms.string('GEMEMapRcd'),
                                                                tag = cms.string('GEMEMap_v2')
                                                                )
                                                      ),
                                   )

process.reader = cms.EDAnalyzer( "GEMEMapDBReader" )

process.recordDataGetter = cms.EDAnalyzer( "EventSetupRecordDataGetter",
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

#Path definition
process.GEMEMapSourcePath = cms.Path( process.reader + process.recordDataGetter )
process.esout = cms.EndPath( process.escontent + process.esretrieval )

#Schedule definition
process.schedule = cms.Schedule( process.GEMEMapSourcePath,
                                 process.esout
                                 )

for name, module in process.es_sources_().iteritems():
    print "ESModules> provider:%s '%s'" % ( name, module.type_() )
for name, module in process.es_producers_().iteritems():
    print "ESModules> provider:%s '%s'" % ( name, module.type_() )
