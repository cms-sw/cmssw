import FWCore.ParameterSet.Config as cms

process = cms.Process("dqm")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("HLTriggerOffline.Egamma.EgammaValidation_cff")
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation")                   
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0006/32CE4A5C-A177-DE11-A597-001D09F2B2CF.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0006/2EB16C72-8F77-DE11-B2D4-001D09F23A6B.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/ECACB3FF-5677-DE11-8431-001D09F241D2.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/EA101CB8-5677-DE11-836B-001D09F24DDF.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/E0F1C645-5677-DE11-92A3-001D09F24DA8.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/CAFBCE54-5577-DE11-80D2-000423DD2F34.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/ACC5CB8B-5777-DE11-BDDC-001D09F2447F.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/76ED83B7-5477-DE11-A317-000423D986C4.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/700F289A-5277-DE11-B623-001D09F24600.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/5052BF40-5677-DE11-A70C-001D09F24047.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/488A68F3-5177-DE11-B924-000423D99658.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/462E7C42-5777-DE11-B001-000423D99E46.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/38199B15-5B77-DE11-926B-0019B9F72BAA.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/1633FFA1-5977-DE11-908D-000423D98B28.root',
                '/store/relval/CMSSW_3_2_1/RelValGammaJets_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0005/0A8157C8-5777-DE11-869D-001D09F28C1E.root'
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
