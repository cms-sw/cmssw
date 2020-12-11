# to test the communication with DBS and produce the csctf configuration
import FWCore.ParameterSet.Config as cms

process = cms.Process("QWE")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource")

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('tscKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "TSC key")
options.register('rsKey',
                 'dummy', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "RS key")
options.register('DBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "DB authenification token path")
options.parseArguments()

process.load("CondTools.L1TriggerExt.L1TriggerKeyRcdSourceExt_cfi")
process.load("CondTools.L1TriggerExt.L1SubsystemKeysOnlineExt_cfi")
process.L1SubsystemKeysOnlineExt.tscKey = cms.string( options.tscKey )
process.L1SubsystemKeysOnlineExt.rsKey  = cms.string( options.rsKey )
process.L1SubsystemKeysOnlineExt.onlineAuthentication = cms.string( options.DBAuth )
process.L1SubsystemKeysOnlineExt.forceGeneration = cms.bool(True)

process.l1cr = cms.EDAnalyzer( "L1TriggerKeyExtReader", label = cms.string("SubsystemKeysOnly") )
process.p = cms.Path(process.l1cr)
