import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)


process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring(
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/0CB7ECDC-7C6C-DD11-A2A8-001617E30CD4.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/1A487B13-7D6C-DD11-9191-001617DBD316.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/24E32910-7D6C-DD11-97E2-000423D94990.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/2AF4810F-7D6C-DD11-BFAA-0016177CA7A0.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/482D641C-7D6C-DD11-8D71-0019DB29C614.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/6645562B-7D6C-DD11-B969-000423D9970C.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/946D15DF-7C6C-DD11-B3B5-000423D9939C.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/9620BF16-7D6C-DD11-92D6-000423D94A20.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/A2A39D12-7D6C-DD11-9B42-001617C3B6CC.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/A8B48C11-7D6C-DD11-A1BC-000423D98844.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/B41F6CB6-7C6C-DD11-B078-000423D98B08.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/CE614936-7D6C-DD11-A010-001617E30CE8.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/D004F8E2-7C6C-DD11-B6D1-001617C3B76A.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/EEC8ACE2-7C6C-DD11-952A-000423D98B5C.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/EEEE040F-7D6C-DD11-B6D9-000423D94E1C.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0005/0ED6B05C-8B6C-DD11-BE33-000423D6BA18.root',
                '/store/relval/CMSSW_2_1_4/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0005/8CADCEB8-AC6C-DD11-8EDD-000423D94700.root'
        
           )
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.DQMStore = cms.Service("DQMStore")

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


#Define the Paths
process.validation = cms.Path(process.HLTTauVal)

process.postProcess = cms.EndPath(process.HLTTauPostVal+process.dqmSaver)

process.schedule = cms.Schedule(process.validation,process.postProcess)












