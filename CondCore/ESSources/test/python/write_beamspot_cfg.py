
from __future__ import print_function
import time

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.AlCa.autoCond import autoCond

options = VarParsing.VarParsing()
options.register('connectionString',
                 'sqlite_file:cms_conditions.db', #default value
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

CondDBParameters = cms.PSet( 
                             messageLevel = cms.untracked.int32( 3 ),
                             )

process.source = cms.Source( "EmptySource",
                             firstRun = cms.untracked.uint32( 1 ),
                             firstLuminosityBlock = cms.untracked.uint32( 1 ),
                             numberEventsInRun = cms.untracked.uint32( 1 ), 
                             numberEventsInLuminosityBlock = cms.untracked.uint32( 1 )
                             )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.prod = cms.EDAnalyzer("LumiTestWriteAnalyzer",
                              connectionString = cms.untracked.string(options.connectionString ),
                              tagName = cms.untracked.string(options.tag),
                              runNumber = cms.untracked.uint32(options.runNumber),
                              numberOfLumis = cms.untracked.uint32(24),
                              iovSize = cms.untracked.uint32(4)
)

process.p = cms.Path( process.prod )

for name, module in process.es_sources_().items():
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
for name, module in process.es_producers_().items():
    print("ESModules> provider:%s '%s'" % ( name, module.type_() ))
