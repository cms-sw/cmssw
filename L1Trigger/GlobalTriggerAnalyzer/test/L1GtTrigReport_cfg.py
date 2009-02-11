#
# cfg file to run on GT output file containing the record L1GlobalTriggerReadoutRecord
# or the lite L1GlobalTriggerRecord
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("RunL1GtTrigReport")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtTrigReport_source.root')
)

#process.PoolSource.fileNames = [
#    '/store/relval/2008/6/25/RelVal-RelValQCD_Pt_120_170-1214239099-STARTUP_V1-2nd/0007/049232E8-AF42-DD11-BC5B-000423D9880C.root',
#    '/store/relval/2008/6/25/RelVal-RelValQCD_Pt_120_170-1214239099-STARTUP_V1-2nd/0007/04A4CD1C-B542-DD11-8ACE-000423D98C20.root',
#    '/store/relval/2008/6/25/RelVal-RelValQCD_Pt_120_170-1214239099-STARTUP_V1-2nd/0007/0AF90409-AF42-DD11-BA95-001617DBCF6A.root'
#]

# /RelValQCD_Pt_80_120/CMSSW_2_2_0_IDEAL_V9_v3/GEN-SIM-DIGI-RAW-HLTDEBUG
process.PoolSource.fileNames = [
    '/store/relval/CMSSW_2_2_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v3/0000/0065D75A-15BA-DD11-A900-001617E30D0A.root',
    '/store/relval/CMSSW_2_2_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v3/0000/1A6B7863-14BA-DD11-83AB-0019DB29C5FC.root',
    '/store/relval/CMSSW_2_2_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v3/0000/2CA699A4-14BA-DD11-907C-001617C3B6FE.root',
    '/store/relval/CMSSW_2_2_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v3/0000/300C86FF-12BA-DD11-9F6D-001617C3B65A.root'
]

# load and configure modules

# L1 GT EventSetup
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

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

# this module
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
 
#process.l1GtTrigReport.PrintVerbosity = 10
#process.l1GtTrigReport.PrintOutput = 1
    
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = True
process.l1GtTrigReport.L1GtRecordInputTag = 'simGtDigis'

# path to be run
process.p = cms.Path(process.l1GtTrigReport)

# services

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules = ['l1GtTrigReport']
process.MessageLogger.destinations = ['testGt_L1GtTrigReport']

