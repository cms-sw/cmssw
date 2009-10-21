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
   fileNames = cms.untracked.vstring(        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/E8719AA8-2832-DE11-8F53-000423D98804.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/E69087B8-2732-DE11-8196-001617C3B6C6.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/E67B0B8A-2832-DE11-A2D6-000423D98E54.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/E0F3533F-2732-DE11-8CEE-000423D98800.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/DEF8B656-2732-DE11-8063-000423D95030.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/DE843C3F-2832-DE11-85C1-000423D94AA8.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/DE5F1F95-2832-DE11-BDD8-000423D8FA38.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/C49F3487-2832-DE11-98DF-001617C3B6E2.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/A8C5B97F-2832-DE11-8B52-0019DB2F3F9A.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/9E4A9435-2732-DE11-8743-000423D98750.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/8C104D08-2932-DE11-A338-000423D98804.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/800C8CA7-2832-DE11-ACBB-000423D99996.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/7EDBAB48-2732-DE11-962C-000423D6B444.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/7A0D1781-2832-DE11-B9B8-001617DBD472.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/70946385-2832-DE11-8B39-000423D952C0.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/56ADF1AF-4D32-DE11-B20C-000423D98B28.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/4AAA10A1-2932-DE11-91F0-000423D98804.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/48993BA3-2832-DE11-9BD9-000423D98EA8.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/367F1E97-2832-DE11-9757-001617DBCF1E.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/2213F58B-2832-DE11-8E5A-000423D98BC4.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/1CFDB8BB-2832-DE11-81CB-000423D6C8E6.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/1C63247C-2832-DE11-B574-001617E30D40.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/14FD7287-2832-DE11-BD79-000423D94700.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/0ED95E48-2832-DE11-AC95-000423D944FC.root',
        '/store/relval/CMSSW_2_2_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0001/0CF388B0-2832-DE11-A1DF-000423D9870C.root')
)
