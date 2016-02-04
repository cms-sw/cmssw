#
# test_RawToDigi_RelVal.py
#
# Test 1 : run unpacker on RelVal raw file and compare output with original emulator digis
#

import FWCore.ParameterSet.Config as cms

process = cms.Process('testRawToDigiRelVal')

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 5000 ) )

# GCT Unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.inputLabel = cms.InputTag( "rawDataCollector" )

# comparator
process.load('L1Trigger.HardwareValidation.L1Comparator_cfi')
process.l1compare.GCTsourceData = cms.InputTag( "l1GctHwDigis" )
process.l1compare.GCTsourceEmul = cms.InputTag( "simGctDigis" )
process.l1compare.VerboseFlag = cms.untracked.int32(0)
process.l1compare.DumpMode = cms.untracked.int32(0) #was -1 (shows failed + worked) or 1 (shows failed only)
process.l1compare.DumpFile = cms.untracked.string( "l1compare_dump.txt" )
process.l1compare.COMPARE_COLLS = cms.untracked.vuint32(
# ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
)

# GCT EXPERT EMU DQM
process.load('DQMServices.Core.DQM_cfg')
process.load('DQM.L1TMonitor.L1TdeGCT_cfi')
process.l1demongct.VerboseFlag = cms.untracked.int32(0)
process.l1demongct.DataEmulCompareSource = cms.InputTag("l1compare")
process.l1demongct.HistFile = cms.untracked.string('test_RawToDigi_RelVal.root')
process.l1demongct.disableROOToutput = cms.untracked.bool( False )

process.defaultPath = cms.Sequence (
process.l1GctHwDigis *
process.l1compare * 
process.l1demongct)

process.p = cms.Path(process.defaultPath)

# RelVal input sample
process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/FE9F3CF9-00D2-DE11-A496-001D09F295A1.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/ECF4C4A0-01D2-DE11-9A02-001D09F28F25.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/D43FBDEB-03D2-DE11-841C-001D09F2960F.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/CE50334E-02D2-DE11-B06E-001D09F291D7.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/C8B63DBD-00D2-DE11-B86A-001D09F232B9.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/A670DB12-02D2-DE11-93AC-001D09F2423B.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/A6264B7E-02D2-DE11-846B-003048D374F2.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/90D48228-01D2-DE11-9F10-000423D94E1C.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/8E47E274-02D2-DE11-84C3-001D09F241B9.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/8AF1A4F1-03D2-DE11-91F1-0019B9F581C9.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/7E023C03-01D2-DE11-A252-003048D2C0F4.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/7AF2F6AA-02D2-DE11-80B4-000423D6B48C.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/74224C8A-03D2-DE11-840D-00304879FBB2.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/7070017B-04D2-DE11-A2A2-001D09F2546F.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/6AD40188-02D2-DE11-B606-001D09F29169.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/486D81E3-1ED2-DE11-8422-000423D9997E.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/4485965D-03D2-DE11-886E-0030486733D8.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/42042775-01D2-DE11-A999-001D09F2AD84.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/3E82B179-02D2-DE11-90E2-001D09F24664.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/3C3D89A5-04D2-DE11-91DD-001D09F252DA.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/3AE600BB-02D2-DE11-AB8D-001D09F34488.root',
              '/store/relval/CMSSW_3_3_3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0003/307173F4-02D2-DE11-823A-001D09F24303.root'
       )
)


