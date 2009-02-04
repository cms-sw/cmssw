#
# cfg file to run L1 Global Trigger emulator on a file containing the output of the 
# GCT system and GMT system, followed by the L1 trigger report
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("RunL1GtEmulTrigReport")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_Emulator_GctGmtFile_source.root'),
    secondaryFileNames = cms.untracked.vstring() 
)

# /RelValQCD_Pt_80_120/CMSSW_2_1_8_IDEAL_V9_v1/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO
#process.PoolSource.fileNames = [
#       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0003/00ABE731-9382-DD11-9605-001D09F28D4A.root',
#       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0003/027E628C-9382-DD11-A912-001617DBD472.root',
#       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0003/0407E1C3-9382-DD11-92B9-001617C3B710.root',
#       '/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0003/068C57CA-9E82-DD11-9186-001617C3B6DE.root',
#]

# /RelValQCD_Pt_80_120/CMSSW_2_1_10_IDEAL_V9_v2/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO
#process.PoolSource.fileNames = [
#       '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/046AC296-EC99-DD11-9691-000423D6A6F4.root',
#       '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/08CF0B41-E599-DD11-B957-000423D98AF0.root',
#       '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/10469277-E099-DD11-9F4C-000423D9890C.root',
#       '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/1C633A33-EA99-DD11-906A-001617DC1F70.root',
#]

# /RelValQCD_Pt_80_120/CMSSW_2_2_0_pre1_IDEAL_V9_v2/GEN-SIM-DIGI-RAW-HLTDEBUG
process.PoolSource.fileNames = [
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/043A4FB0-A4B1-DD11-B4AA-000423D99660.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/1EF31EC5-A6B1-DD11-A94D-000423D9989E.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/22362FB1-A4B1-DD11-88AB-000423D987E0.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/323C28D4-A4B1-DD11-936D-000423D99B3E.root',
]

# load and configure modules

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_V9::All"

# prescaled menu    
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_cff")

# uprescaled menu - change prescale factors to 1
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_Unprescaled_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_Unprescaled_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_Unprescaled_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1MenuTestCondCorrelation_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v4_Unprescaled_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v5_Unprescaled_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v1_Unprescaled_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")


# WARNING: use always the same prescale factors and trigger mask for data/emulator
#          and this module! Safer is to run emulator and report in one step


# Global Trigger emulator
import L1Trigger.GlobalTrigger.gtDigis_cfi
process.l1GtEmulDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()

# input tag for GMT readout collection: 
#     gmtDigis = GMT emulator (default)
#     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
process.l1GtEmulDigis.GmtInputTag = 'gtDigis'

# input tag for GCT readout collections: 
#     gctDigis = GCT emulator (default) 
process.l1GtEmulDigis.GctInputTag = 'gctDigis'

# logical flag to produce the L1 GT DAQ readout record
#     if true, produce the record (default)
#process.l1GtEmulDigis.ProduceL1GtDaqRecord = False
    
# logical flag to produce the L1 GT EVM readout record
#     if true, produce the record (default)
process.l1GtEmulDigis.ProduceL1GtEvmRecord = False

# logical flag to produce the L1 GT object map record
#     if true, produce the record (default)
process.l1GtEmulDigis.ProduceL1GtObjectMapRecord = False

# logical flag to write the PSB content in the  L1 GT DAQ record
#     if true, write the PSB content in the record (default)
#process.l1GtEmulDigis.WritePsbL1GtDaqRecord = False

# logical flag to read the technical trigger records
#     if true, it will read via getMany the available records (default)
#process.l1GtEmulDigis.ReadTechnicalTriggerRecords = False

# number of "bunch crossing in the event" (BxInEvent) to be emulated
# symmetric around L1Accept (BxInEvent = 0):
#    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
# even numbers (except 0) "rounded" to the nearest lower odd number
# negative value: emulate TotalBxInEvent as given in EventSetup  
process.l1GtEmulDigis.EmulateBxInEvent = 1

# length of BST record (in bytes) from parameter set
# negative value: take the value from EventSetup      
#process.l1GtEmulDigis.BstLengthBytes = 52

#
# l1GtTrigReport module
#

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
 
# boolean flag to select the input record
# if true, it will use L1GlobalTriggerRecord 
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

# input tag for GT record: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
process.l1GtTrigReport.L1GtRecordInputTag = "l1GtEmulDigis"

#process.l1GtTrigReport.PrintVerbosity = 10
#process.l1GtTrigReport.PrintOutput = 1
    
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = true
#process.l1GtTrigReport.L1GtRecordInputTag = l1GtRecord

# path to be run
process.p = cms.Path(process.l1GtEmulDigis*process.l1GtTrigReport)

# services

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules = ['l1GtTrigReport']
process.MessageLogger.destinations = ['L1GtEmulTrigReport']

