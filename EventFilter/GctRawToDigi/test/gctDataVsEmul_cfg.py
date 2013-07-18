import FWCore.ParameterSet.Config as cms

process = cms.Process('testAnalysis')

#Logger thingy
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service ("MessageLogger", 
  destinations = cms.untracked.vstring( "detailedInfo.txt" ),
  threshold = cms.untracked.string ( 'WARNING' )
)

process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring('/store/data/Commissioning08/Calo/RAW/v1/000/066/279/000B8879-AB9A-DD11-80EE-001617C3B778.root') 
)

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 5 ) )

# GCT Unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.inputLabel = cms.InputTag( "rawDataCollector" )
process.l1GctHwDigis.verbose = cms.untracked.bool ( False )
process.l1GctHwDigis.unpackFibres = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternEm = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternJets = cms.untracked.bool ( True )

# GCT emulator
process.load('L1Trigger.Configuration.L1StartupConfig_cff')
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.inputLabel = cms.InputTag( "l1GctHwDigis" )#use this when running over random trigger data
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)

# comparator
process.load('L1Trigger.HardwareValidation.L1Comparator_cfi')
process.l1compare.GCTsourceData = cms.InputTag( "l1GctHwDigis" )
process.l1compare.GCTsourceEmul = cms.InputTag( "valGctDigis" )
#process.l1compare.RCTsourceData = cms.InputTag( "l1GctHwDigis" )
#process.l1compare.RCTsourceEmul = cms.InputTag( "l1GctHwDigis" )
process.l1compare.VerboseFlag = cms.untracked.int32(0)
process.l1compare.DumpMode = cms.untracked.int32(0) #was -1 (shows failed + worked) or 1 (shows failed only)
process.l1compare.DumpFile = cms.untracked.string( "l1compare_dump.txt" )
process.l1compare.COMPARE_COLLS = cms.untracked.vuint32(
# ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
)

# GCT DQM
process.load('DQMServices.Core.DQM_cfg')
process.load('DQM.L1TMonitor.L1TGCT_cfi')
process.l1tgct.disableROOToutput = cms.untracked.bool(False)
process.l1tgct.outputFile = cms.untracked.string('gctDqm_testAnalysis.root')
process.l1tgct.gctCentralJetsSource = cms.InputTag("l1GctHwDigis","cenJets")
process.l1tgct.gctNonIsoEmSource = cms.InputTag("l1GctHwDigis","nonIsoEm")
process.l1tgct.gctForwardJetsSource = cms.InputTag("l1GctHwDigis","forJets")
process.l1tgct.gctIsoEmSource = cms.InputTag("l1GctHwDigis","isoEm")
process.l1tgct.gctEnergySumsSource = cms.InputTag("l1GctHwDigis","")
process.l1tgct.gctTauJetsSource = cms.InputTag("l1GctHwDigis","tauJets")

# RCT DQM
process.load('DQM.L1TMonitor.L1TRCT_cfi')
process.l1trct.disableROOToutput = cms.untracked.bool(False)
process.l1trct.outputFile = cms.untracked.string('gctDqm_testAnalysis.root')
process.l1trct.rctSource = cms.InputTag("l1GctHwDigis","")
#process.l1trct.rctSource = cms.InputTag("valGctDigis","")

# GCT EXPERT EMU DQM
process.load('DQM.L1TMonitor.L1TdeGCT_cfi')
process.l1demongct.VerboseFlag = cms.untracked.int32(0)
process.l1demongct.DataEmulCompareSource = cms.InputTag("l1compare")
process.l1demongct.HistFile = cms.untracked.string('gctDqm_testAnalysis.root')
process.l1demongct.disableROOToutput = cms.untracked.bool( False )

process.defaultPath = cms.Sequence (
process.l1GctHwDigis *
process.valGctDigis * 
process.l1compare * 
process.l1trct * 
process.l1tgct * 
process.l1demongct)

process.p = cms.Path(process.defaultPath)

process.output = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring (
  "drop *",
  "keep *_*_*_testAnalysis"  
  ),
  fileName = cms.untracked.string( "run68288_10M_V01-15-16.root" ) #/castor/cern.ch/user/j/jad/run68288_09val/
)

#process.out = cms.EndPath( process.output )
