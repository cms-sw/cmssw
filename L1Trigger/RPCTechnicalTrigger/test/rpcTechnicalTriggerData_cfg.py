import FWCore.ParameterSet.Config as cms
import os

maxevts   = 100
globaltag = 'GR09_31X_V4P::All'
globaltag = 'STARTUP3XY_V9::All'
inputfile = '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/468/F2CDA89C-B57D-DE11-B2C9-000423D98DC4.root'

process   = cms.Process("RPCTechnicalTrigger")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.categories = ['*']
process.MessageLogger.destinations = ['cout']
process.MessageLogger.cout = cms.untracked.PSet(
    	threshold = cms.untracked.string('DEBUG'),
	INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(-1) ) )

#.. Geometry and Global Tags
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string( globaltag )
process.load("Configuration.StandardSequences.MagneticField_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxevts) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( inputfile ) )

process.load("L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi")

#.. use the provided hardware configuration parameters
process.rpcTechnicalTrigger.UseDatabase   = cms.untracked.int32(0)
process.rpcTechnicalTrigger.ConfigFile    = cms.untracked.string("hardware-pseudoconfig.txt")
process.rpcTechnicalTrigger.RPCDigiLabel  = cms.InputTag("muonRPCDigis")

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('rpcttbits.root'),
                               outputCommands = cms.untracked.vstring('drop *','keep L1GtTechnicalTriggerRecord_*_*_*') )

process.p = cms.Path(process.rpcTechnicalTrigger)

process.e = cms.EndPath(process.out)

