import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("test")

process.MessageLogger = cms.Service( "MessageLogger",
                                     debugModules = cms.untracked.vstring( "*" ),
                                     cout = cms.untracked.PSet( threshold = cms.untracked.string( "DEBUG" ) ),
                                     destinations = cms.untracked.vstring( "cout" )
                                     )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1 ) )

process.source = cms.Source( "EmptySource",
                             numberEventsInRun = cms.untracked.uint32(1),
                             firstRun = cms.untracked.uint32(1)
                             )

process.load("CondCore.CondDB.CondDB_cfi")
process.siStripO2O = cms.EDAnalyzer( "SiStripPayloadMapTableCreator",
                                     process.CondDB,
#                                      configMapDatabase = cms.string("sqlite:configMap.db"),
#                                      configMapDatabase = cms.string("oracle://cms_orcoff_prep/CMS_COND_STRIP"),
                                     configMapDatabase = cms.string("oracle://cms_orcon_prod/CMS_COND_O2O"),
                                     )

process.p = cms.Path(process.siStripO2O)
