import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTVAL")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/00C9C666-B3B2-DD11-A3E5-000423D9853C.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/0AAE2E10-B3B2-DD11-AC65-001617E30CC8.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/0C25E503-B3B2-DD11-8BF3-000423D98EA8.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/1469C60D-B3B2-DD11-BF43-000423D99F3E.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/16FCDE64-B3B2-DD11-B009-000423D6B42C.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/223CA255-B3B2-DD11-A293-000423D6C8E6.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/441F935A-B3B2-DD11-8765-000423D33970.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/4ADA9A55-B3B2-DD11-96DD-000423D991D4.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/6AC3F45B-B3B2-DD11-B891-001617C3B6DE.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/820D8F0F-B3B2-DD11-9EA4-000423D9997E.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/AA6D155B-B3B2-DD11-B75E-000423D99F3E.root',
                '/store/relval/CMSSW_3_0_0_pre2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v2/0001/FE41EC0B-B3B2-DD11-A9E1-000423D9A212.root'
        
    )
)


#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")


#Reconfigure Environment and saver
#process.dqmEnv.subSystemFolder = cms.untracked.string('HLT/HLTTAU')
#process.DQM.collectorPort = 9091
#process.DQM.collectorHost = cms.untracked.string('pcwiscms10')

process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/A/N/C')
process.dqmSaver.forceRunNumber = cms.untracked.int32(123)


#Load the Validation
process.load("HLTriggerOffline.Tau.Validation.HLTTauValidation_cff")

#Load The Post processor
process.load("HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi")
process.load("HLTriggerOffline.Tau.Validation.HLTTauQualityTests_cff")


#Define the Paths
process.validation = cms.Path(process.HLTTauVal)

process.postProcess = cms.EndPath(process.HLTTauPostVal+process.hltTauRelvalQualityTests+process.dqmSaver)
#process.postProcess = cms.EndPath(process.dqmSaver)
process.schedule =cms.Schedule(process.validation,process.postProcess)



