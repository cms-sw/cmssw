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
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/FA7998E0-3999-DE11-ABD0-00304867920C.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/FA6C82EF-3999-DE11-89FA-001731AF6BD3.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F8982D55-2299-DE11-B6D6-001731AF687F.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F6A44E75-2299-DE11-9537-0018F3D09678.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F67D8909-CB98-DE11-8A97-003048678B86.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F42FFA84-CA98-DE11-AB82-0018F3D0961A.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/F296F67F-2299-DE11-B414-001A92971B38.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/EED22D89-2299-DE11-9CBD-001A92810AEA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/EE0E8370-2299-DE11-8FD0-0018F3D0967E.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/EAD24A70-2299-DE11-A577-0018F3D0961A.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/E6EE64EF-3999-DE11-B049-001731AF66AF.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/E6D76B53-2299-DE11-B42D-001731AF6BC1.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/DEDBFB81-9F98-DE11-94B8-0018F3D0967E.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/DCFADA7F-2299-DE11-8B70-001731AF66EF.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D8CE2597-2299-DE11-94FE-0018F3D096E6.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D892C2B1-CA98-DE11-A5A9-001A92971AA8.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D88E3C61-2299-DE11-824F-0018F3D096BA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D4FF6838-CB98-DE11-BE10-0017312310E7.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D452C271-9F98-DE11-A96F-001731AF6765.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D2056A6D-2299-DE11-80C6-001A92971BDA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/D0BC3269-9F98-DE11-B240-0018F3D096CA.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/CCF49F7B-2299-DE11-9256-0018F3D0966C.root',
        '/store/data/CRAFT09/Calo/RAW-RECO/GR09_31X_V5P_HCALHighEnergy-332_v4/0022/CAE32973-2299-DE11-AE58-001A92810AEE.root'
        )
)

