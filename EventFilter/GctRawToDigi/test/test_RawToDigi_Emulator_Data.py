#
# test_RawToDigi_Emulator_Data.py
#
# Test 3 : Run unpacker on raw data and compare with emulator
#


import FWCore.ParameterSet.Config as cms

process = cms.Process('testRawToDigiEmulatorData')

#Logger thingy
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service ("MessageLogger", 
#  destinations = cms.untracked.vstring( "detailedInfo.txt" ),
#  threshold = cms.untracked.string ( 'WARNING' )
#)

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( -1 ) )

# GCT Unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.inputLabel = cms.InputTag( "source" )
process.l1GctHwDigis.verbose = cms.untracked.bool ( False )
process.l1GctHwDigis.unpackFibres = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternEm = cms.untracked.bool ( True )
process.l1GctHwDigis.unpackInternJets = cms.untracked.bool ( True )

# GCT emulator
process.load('L1Trigger.Configuration.L1StartupConfig_cff')
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
process.valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
process.valGctDigis.inputLabel = cms.InputTag( "l1GctHwDigis" )
process.valGctDigis.preSamples = cms.uint32(0)
process.valGctDigis.postSamples = cms.uint32(0)

# comparator
process.load('L1Trigger.HardwareValidation.L1Comparator_cfi')
process.l1compare.GCTsourceData = cms.InputTag( "l1GctHwDigis" )
process.l1compare.GCTsourceEmul = cms.InputTag( "valGctDigis" )
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
process.l1trct.outputFile = cms.untracked.string('test_RawToDigi_Emulator_Data.root')
process.l1trct.rctSource = cms.InputTag("l1GctHwDigis","")


# GCT EXPERT EMU DQM
process.load('DQM.L1TMonitor.L1TdeGCT_cfi')
process.l1demongct.VerboseFlag = cms.untracked.int32(0)
process.l1demongct.DataEmulCompareSource = cms.InputTag("l1compare")
process.l1demongct.HistFile = cms.untracked.string('test_RawToDigi_Emulator_Data.root')
process.l1demongct.disableROOToutput = cms.untracked.bool( False )

process.defaultPath = cms.Sequence (
process.l1GctHwDigis *
process.valGctDigis * 
process.l1compare * 
process.l1trct * 
process.l1tgct * 
process.l1demongct)

process.p = cms.Path(process.defaultPath)

process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/FA4886DB-7FAB-DD11-95F4-000423D98BC4.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/FA30500C-89AB-DD11-B100-000423D99AAE.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/F8BF1FBB-7DAB-DD11-AE60-001617E30D06.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/F42665F5-81AB-DD11-8346-000423D6AF24.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/F2E0F9E8-78AB-DD11-8164-001617C3B778.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/EA691726-7FAB-DD11-812B-001617E30D06.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/EA1D3022-7FAB-DD11-A103-000423D6B2D8.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/E8A7682D-86AB-DD11-8346-000423D94990.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/E6F3E436-7AAB-DD11-A381-000423D9517C.root',
                '/store/data/Commissioning08/Calo/RAW/v1/000/069/522/E6A50648-81AB-DD11-9898-000423D6C8EE.root'
        )
)

