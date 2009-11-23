import FWCore.ParameterSet.Config as cms

process = cms.Process('GctVmeDoAll')

process.source = cms.Source ( "EmptySource" )

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(100)


# N events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 3563 ) )

# raw data
process.gctRaw = cms.OutputModule( "TextToRaw",
  filename = cms.untracked.string("patternCapture_ts__2009_11_03__17h21m43s.txt"),
  GctFedId = cms.untracked.int32 ( 745 )
)

# dump raw
process.dumpRaw = cms.OutputModule ( "DumpFEDRawDataProduct",
  feds = cms.untracked.vint32 ( 745 ),
  dumpPayload = cms.untracked.bool ( True )
)
  
# digis
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.inputLabel = cms.InputTag( "gctRaw" )
process.l1GctHwDigis.verbose = cms.untracked.bool ( True )
process.l1GctHwDigis.numberOfGctSamplesToUnpack = cms.uint32(1)
process.l1GctHwDigis.numberOfRctSamplesToUnpack = cms.uint32(1)

# dump digis
process.load('L1Trigger.L1GctAnalyzer.dumpGctDigis_cfi')
process.dumpGctDigis.rawInput = cms.untracked.InputTag( "l1GctHwDigis" )
process.dumpGctDigis.emuRctInput = cms.untracked.InputTag( "rctDigis" )
process.dumpGctDigis.emuGctInput = cms.untracked.InputTag( "gctDigis" )
process.dumpGctDigis.doHardware = cms.untracked.bool ( True )
process.dumpGctDigis.doEmulated = cms.untracked.bool ( False )
process.dumpGctDigis.doRctEm = cms.untracked.bool ( True )
process.dumpGctDigis.doInternEm = cms.untracked.bool ( False )
process.dumpGctDigis.doEm = cms.untracked.bool ( True )
process.dumpGctDigis.doJets = cms.untracked.bool ( True )
process.dumpGctDigis.doEnergySums = cms.untracked.bool ( True )

# emulator
process.load('L1Trigger.Configuration.L1DummyConfig_cff')
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.inputLabel = 'l1GctHwDigis'

# comparator
process.load('L1Trigger.HardwareValidation.L1Comparator_cfi')
process.l1compare.COMPARE_COLLS = cms.untracked.vuint32(
# ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
)

# GCT DQM
process.load('DQMServices.Core.DQM_cfg')

process.load('DQM.L1TMonitor.L1TGCT_cfi')
process.l1tgct.disableROOToutput = cms.untracked.bool(False)
process.l1tgct.outputFile = cms.untracked.string('gctDqm.root')
process.l1tgct.gctCentralJetsSource = cms.InputTag("l1GctHwDigis","cenJets")
process.l1tgct.gctNonIsoEmSource = cms.InputTag("l1GctHwDigis","nonIsoEm")
process.l1tgct.gctForwardJetsSource = cms.InputTag("l1GctHwDigis","forJets")
process.l1tgct.gctIsoEmSource = cms.InputTag("l1GctHwDigis","isoEm")
process.l1tgct.gctEnergySumsSource = cms.InputTag("l1GctHwDigis","")
process.l1tgct.gctTauJetsSource = cms.InputTag("l1GctHwDigis","tauJets")

# RCT DQM
process.load('DQM.L1TMonitor.L1TRCT_cfi')
process.l1trct.disableROOToutput = cms.untracked.bool(False)
#process.l1trct.outputFile = cms.untracked.string('gctDqm.root')
process.l1trct.rctSource = cms.InputTag("l1GctHwDigis","")

# EMU DQM
process.load('DQM.L1TMonitor.L1TDEMON_cfi')
process.l1demon.VerboseFlag = cms.untracked.int32(1)
process.l1demon.HistFile = cms.untracked.string('gctDqm.root')

process.defaultPath = cms.Sequence ( process.gctRaw * process.l1GctHwDigis
# print Raw
#                       * process.dumpRaw
# print GCT digis
#                       * process.dumpGctDigis
# RCT DQM
                       * process.l1trct
# GCT DQM
                       * process.l1tgct
# Emulator DQM
                       * process.valGctDigis * process.l1compare * process.l1demon )

process.p = cms.Path(process.defaultPath)

# Write out digis in file
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("gctVmeDoAll.root")
    )

process.outpath=cms.EndPath(process.out)

