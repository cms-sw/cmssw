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

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

#readFiles.extend( ('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_Analyzer_source.root') );

# /RelValQCD_Pt_80_120/CMSSW_2_2_0_pre1_IDEAL_V9_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
readFiles.extend( ( 
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/0C307457-92AE-DD11-BD8F-000423D94700.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/20FEDE7C-93AE-DD11-BD75-000423D94700.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/28109DCE-91AE-DD11-AB88-001617E30D0A.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/2CEE9A51-93AE-DD11-B7D6-001617DBD472.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/3C19E04B-91AE-DD11-9D31-000423D991D4.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/44414E66-92AE-DD11-9864-001617E30CE8.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/52CF36A1-91AE-DD11-A5A2-000423D9853C.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/568705A2-92AE-DD11-905E-001617E30CA4.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/58668BE5-92AE-DD11-97BC-001617C3B654.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/5EEE70CC-92AE-DD11-8BB9-001617E30CA4.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/6CD5E7F7-91AE-DD11-97B6-001617C3B706.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/7A71C98D-93AE-DD11-B860-001617E30CA4.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/8E2F8524-92AE-DD11-8FFA-001D09F24EE3.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/90C2A509-93AE-DD11-8892-0016177CA778.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/B41A163B-93AE-DD11-BB6A-000423D94534.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/CA02E2FD-92AE-DD11-A47A-000423D985B0.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/D8CE2077-92AE-DD11-872C-001617C3B6DC.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/F24E4C20-92AE-DD11-9EEA-000423D985E4.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/F25FE255-92AE-DD11-80BD-000423D98FBC.root',
       '/store/relval/CMSSW_2_2_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/F6E325B2-92AE-DD11-B98A-000423D95030.root') );


secFiles.extend( (
               ) )

# load and configure modules

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

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtAnalyz_cfi")

# input tag for GT readout collection: 
process.l1GtAnalyz.DaqGtInputTag = 'simGtDigis' 
#process.l1GtAnalyz.DaqGtInputTag = 'l1GtUnpack' 
 
# input tags for GT lite record
#process.l1GtAnalyz.L1GtRecordInputTag = 'l1GtRecord'

# input tag for GT object map collection
process.l1GtAnalyz.GtObjectMapTag = 'hltL1GtObjectMap'

# another algorithm and condition in that algorithm to test the object maps
process.l1GtAnalyz.AlgorithmName = 'L1_SingleEG20'
process.l1GtAnalyz.ConditionName = 'SingleNoIsoEG_0x14'

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

