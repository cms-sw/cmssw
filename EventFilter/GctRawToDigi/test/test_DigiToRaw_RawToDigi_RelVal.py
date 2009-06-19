#
# test_DigiToRaw_RawToDigi_RelVal.py
#
# Test 2 : run packer and unpacker on RelVal, and compare output with original emulator digis
#

import FWCore.ParameterSet.Config as cms

process = cms.Process('testDigiToRawRawToDigiRelVal')

# Number of events
process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 5000 ) )

# GCT packer
process.load('EventFilter.GctRawToDigi.gctDigiToRaw_cfi')

# GCT Unpacker
process.load('EventFilter.GctRawToDigi.l1GctHwDigis_cfi')
process.l1GctHwDigis.inputLabel = cms.InputTag( "gctDigiToRaw" )

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
process.l1demongct.HistFile = cms.untracked.string('test_DigiToRaw_RawToDigi_RelVal.root')
process.l1demongct.disableROOToutput = cms.untracked.bool( False )

process.defaultPath = cms.Sequence (
    process.gctDigiToRaw *
    process.l1GctHwDigis *
    process.l1compare * 
    process.l1demongct
)

process.p = cms.Path(process.defaultPath)

# RelVal input sample
process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E4C24E91-CD57-DE11-90DE-001D09F2532F.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E469F214-C857-DE11-A950-001D09F252E9.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/DE489DF6-C657-DE11-ABCA-001D09F28D54.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/D4CBC4D7-CC57-DE11-924E-001D09F29619.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/CA9880CE-AC57-DE11-B891-001D09F25208.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B67BD00F-F257-DE11-95E5-001D09F2B2CF.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B66C0FD8-CC57-DE11-9593-001D09F297EF.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B0863555-CA57-DE11-B198-000423D94534.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/98DD9243-B357-DE11-82B4-00304879FBB2.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/906F7F83-C557-DE11-BF54-000423D99AAA.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4EC4DD37-CD57-DE11-8F88-0019B9F705A3.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4ADD5FD0-BB57-DE11-B8F0-001617DBD5AC.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4A2A45E1-AA57-DE11-82AA-0030487A1990.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/404DF239-C357-DE11-849D-001D09F241F0.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/3CB939FA-B757-DE11-BFB5-001D09F24448.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/36647208-C757-DE11-94B1-001D09F24763.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/3447A45D-CD57-DE11-9396-001D09F24489.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/124FFE86-CD57-DE11-92D7-001D09F24259.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/124CD9B7-C457-DE11-87D4-001D09F25393.root',
                '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/081FA023-C457-DE11-BAC8-0019B9F704D6.root'    
)
)
