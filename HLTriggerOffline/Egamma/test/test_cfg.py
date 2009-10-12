import FWCore.ParameterSet.Config as cms

process = cms.Process("dqm")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.load("HLTriggerOffline.Egamma.EgammaValidation_cff")
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation")                   
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/10FA04C9-FA15-DE11-9E7F-001617E30D06.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/20B5F1BD-0716-DE11-8081-000423D6C8E6.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/3EB41A00-F315-DE11-8181-000423D985B0.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/3EF3600F-E615-DE11-9B78-000423D9989E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/40B2744C-E815-DE11-8497-000423D99F1E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/481B9795-AB16-DE11-AC5D-001617C3B6FE.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/50776CD8-E915-DE11-984B-000423D9989E.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/52F0FDC6-5216-DE11-8DAC-00161757BF42.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/7C7C96B9-F815-DE11-BAE3-001617C3B6E2.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/823C9B66-E515-DE11-BF44-000423D996C8.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/8840403B-E915-DE11-97EA-001D09F23A02.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/A867B73D-E715-DE11-BC4A-000423D94AA8.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/D85F9459-E915-DE11-A6B2-000423D944F8.root',
                '/store/relval/CMSSW_3_1_0_pre4/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_v1/0003/EA6D953A-E715-DE11-BB29-000423D9989E.root'
                                             )
                            )

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.p = cms.EndPath(process.post+process.dqmSaver)

process.testW = cms.Path(process.egammaValidationSequence)

process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
