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
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/F65E0F8D-2F9A-DE11-B5BC-001D09F251FE.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/E403817C-369A-DE11-823D-001D09F24024.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/DE673AA5-329A-DE11-ADFA-001D09F29597.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/D26D57EB-3E9A-DE11-8553-0030486780B8.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/CE2EFB17-439A-DE11-B0EF-001D09F252E9.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/C841197D-3A9A-DE11-8FA5-001D09F2AD7F.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/C6DB5A18-2F9A-DE11-85A8-001D09F251FE.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/BC62E5A7-429A-DE11-BCE7-001D09F2960F.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/BABCE7BB-FB9A-DE11-AF8B-001D09F2932B.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/BA633229-439A-DE11-B185-001D09F24FEC.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/A0CD2787-409A-DE11-B4B2-001D09F252E9.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/96BADD45-3E9A-DE11-A9FD-000423D6B48C.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/905BB726-399A-DE11-ABE1-001D09F29524.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/7870464A-3E9A-DE11-9BA6-000423D6CA6E.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/5E674943-429A-DE11-9883-003048D37580.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/56C2EAA3-429A-DE11-9CF1-001D09F27067.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/560A0A5B-429A-DE11-9CAD-003048D2C1C4.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/3C6E899C-419A-DE11-B84C-000423D98950.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/1C201D08-339A-DE11-80C0-001D09F29597.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/12257497-3D9A-DE11-A10C-001617C3B65A.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/08D0F1AA-339A-DE11-93CD-003048D2BF1C.root',
        '/store/relval/CMSSW_3_2_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V7-v1/0013/0837007D-329A-DE11-9B1F-001D09F29597.root'
)
)
