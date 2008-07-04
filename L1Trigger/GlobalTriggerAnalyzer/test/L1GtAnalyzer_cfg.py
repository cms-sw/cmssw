#
# cfg file to run on GMT + GCT + GT output file
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("TestGtAnalyzer")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_Analyzer_source.root')
)

#process.PoolSource.fileNames = [
#    '/store/relval/2008/6/25/RelVal-RelValQCD_Pt_120_170-1214239099-STARTUP_V1-2nd/0007/049232E8-AF42-DD11-BC5B-000423D9880C.root',
#    '/store/relval/2008/6/25/RelVal-RelValQCD_Pt_120_170-1214239099-STARTUP_V1-2nd/0007/04A4CD1C-B542-DD11-8ACE-000423D98C20.root',
#    '/store/relval/2008/6/25/RelVal-RelValQCD_Pt_120_170-1214239099-STARTUP_V1-2nd/0007/0AF90409-AF42-DD11-BA95-001617DBCF6A.root'
#]

# load and configure modules

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1Menu2007_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu2008_2E31_cff")
#process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1x1032.L1MenuTestCondCorrelation_cff")

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtAnalyz_cfi")

# input tag for GT readout collection: 
process.l1GtAnalyz.DaqGtInputTag = 'hltGtDigis' 
#process.l1GtAnalyz.DaqGtInputTag = 'l1GtUnpack' 
 
# input tags for GT lite record
#process.l1GtAnalyz.L1GtRecordInputTag = 'l1GtRecord'

# input tag for GT object map collection
process.l1GtAnalyz.GtObjectMapTag = 'hltL1GtObjectMap'

# another algorithm and condition in that algorithm to test the object maps
process.l1GtAnalyz.AlgorithmName = 'L1_SingleEG20'
process.l1GtAnalyz.ConditionName = 'SingleNoIsoEG20'
#process.l1GtAnalyz.AlgorithmName = 'L1_Mu3_Jet15' 
#process.l1GtAnalyz.ConditionName = 'SingleMu3_3' 

# path to be run
process.p = cms.Path(process.l1GtAnalyz)

# services

# uncomment / comment messages with DEBUG mode to run in DEBUG mode
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('testGt_Analyzer'),
    testGt_Analyzer = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'), ## DEBUG mode 

        DEBUG = cms.untracked.PSet( 

            limit = cms.untracked.int32(-1)          ## DEBUG mode, all messages  
            #limit = cms.untracked.int32(10)         ## DEBUG mode, max 10 messages 
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    debugModules = cms.untracked.vstring('l1GtAnalyz'), ## DEBUG mode 
)

# output 
process.outputL1GtAnalyz = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_Analyzer_output.root')
)

process.outpath = cms.EndPath(process.outputL1GtAnalyz)

