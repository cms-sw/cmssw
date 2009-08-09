import FWCore.ParameterSet.Config as cms
import os

maxevts   = 100

inputfile = '/store/relval/CMSSW_3_1_0_pre7/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0004/D0DE109C-E641-DE11-B0CD-001D09F25325.root'

process   = cms.Process("RPCTechnicalTrigger")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.categories = ['*']
process.MessageLogger.destinations = ['cout']
process.MessageLogger.cout = cms.untracked.PSet(
    	threshold = cms.untracked.string('DEBUG'),
	INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(-1) ) )

#.. Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

#.. Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

#.. reconstruction sequence for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

#.. access database hardware configuration objects

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxevts) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( inputfile ) )

process.load("L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi")

process.out = cms.OutputModule("PoolOutputModule",
	                               fileName = cms.untracked.string('rpcttbits.root'),
        	                       outputCommands = cms.untracked.vstring('drop *','keep L1GtTechnicalTriggerRecord_*_*_*') )

process.p = cms.Path(process.rpcTechnicalTrigger)

process.e = cms.EndPath(process.out)

