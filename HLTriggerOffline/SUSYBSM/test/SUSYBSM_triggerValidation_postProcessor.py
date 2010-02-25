import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")
process.load("FWCore.MessageService.MessageLogger_cfi")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")


#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#process.load("Configuration.GlobalRuns.ReconstructionGR_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

#process.load("L1Trigger.Configuration.L1Config_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu2008_2E30_Unprescaled_cff")


#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")

# The GenMET is not in the edm root files. You have to produce it by yourself
process.load("RecoMET.Configuration.GenMETParticles_cff")

process.load("RecoMET.METProducers.genMetTrue_cfi")

#process.load("HLTriggerOffline.SUSYBSM.SUSYBSM_triggerValidation_fastSim_cff")
process.load("HLTriggerOffline.SUSYBSM.SUSYBSM_triggerValidation_cff")
process.load("HLTriggerOffline.SUSYBSM.SUSYBSM_postProcessor_cff")
process.load("HLTriggerOffline.SUSYBSM.HLTSusyExoQualityTester_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0015/34BA4AF3-9A1B-DF11-A66C-003048678FA6.root',
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/D6FD0DFE-551B-DF11-8AB0-001A928116C2.root',
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/92E1636F-531B-DF11-BB63-0018F3D09654.root',
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/50312646-521B-DF11-9FDC-003048D15DCA.root',
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/4A4D7672-561B-DF11-BBBC-001A92811744.root',
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/2E406496-541B-DF11-A329-001731AF678D.root',
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/28311970-531B-DF11-901B-001731AF6725.root',
        '/store/relval/CMSSW_3_5_1/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/1ED9D7F3-541B-DF11-8DA8-001A9281170A.root'

#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0014/5C2F32FF-6C13-DF11-9D99-002354EF3BDD.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/DA73D117-5713-DF11-B4A7-0017313F02F2.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/C85ED17E-5313-DF11-AC6D-001BFCDBD100.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/92EE1C2F-6213-DF11-B579-003048678F74.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/6219BCB0-5513-DF11-8386-0030486792BA.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/48C612EB-5413-DF11-900A-001731AF6721.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/28A5D6F5-5313-DF11-9134-001A92810A98.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/24D66266-5213-DF11-8578-001A92971AD0.root',
#        '/store/relval/CMSSW_3_5_0/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0013/20AACA72-5413-DF11-AD7A-003048D15D22.root'

#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/AA4589C2-2E1E-DF11-B877-0018F3D0968A.root',
#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/76658AE8-2F1E-DF11-B819-0018F3D09704.root',
#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/468CF145-2E1E-DF11-88FB-001A92810AD0.root',
#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/4297BC5E-D91E-DF11-B0AE-003048678B12.root',
#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/38C1B455-2F1E-DF11-B6D4-001A92810A92.root',
#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/3428E9B6-2C1E-DF11-B79A-0030486791BA.root',
#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/2A12D373-3C1E-DF11-9E79-001731AF67B7.root',
#        '/store/relval/CMSSW_3_5_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V21-v1/0016/14872065-3A1E-DF11-BE04-0018F3D0963C.root'


#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/FCF634D2-9813-DF11-A732-001A9281173E.root',
#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/F6BC43EE-8A13-DF11-A104-003048678BE6.root',
#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/ECB7A5E8-8813-DF11-A375-003048679012.root',
#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/72CE10FE-8913-DF11-9B68-003048678BB8.root',
#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/26EC2165-9513-DF11-B381-0026189438C0.root',
#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/1263B078-8B13-DF11-93E0-001A928116F0.root',
#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/08680E0F-9713-DF11-9B0B-001A92971BB2.root',
#        '/store/relval/CMSSW_3_4_2/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v1/0011/0698177E-B413-DF11-AD91-002618943866.root'

#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/FC73FF9D-41E4-DE11-9E16-0018F3D0961A.root',
#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/DE33062D-41E4-DE11-AEA2-003048678B26.root',
#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/A01EDF24-41E4-DE11-BA08-00261894394B.root',
#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/62F1AB1C-42E4-DE11-8044-00304867902C.root',
#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/50B7222A-41E4-DE11-A5BB-003048678B20.root',
#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/4CA02273-40E4-DE11-B4AC-001A92810AEE.root',
#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/4217EFA5-42E4-DE11-8E2F-003048678B20.root',
#        '/store/relval/CMSSW_3_3_6/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V9A-v1/0009/28C4519F-9EE4-DE11-8FE2-0026189438F2.root'

#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/E62DD597-50D5-DE11-9E60-001731A281B1.root',
#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/D4D8799F-6AD5-DE11-A5F1-001A92810ADE.root',
#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/CA11D708-43D5-DE11-88E0-001731AF68C1.root',
#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/C490B780-53D5-DE11-BB4A-001731AF6B7D.root',
#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/A806902B-4ED5-DE11-A7FA-0018F34D0D62.root',
#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/4E463FA7-51D5-DE11-9F5C-0018F34D0D62.root',
#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/483D0642-4FD5-DE11-8642-0018F34D0D62.root',
#        '/store/relval/CMSSW_3_3_4/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V9-v1/0001/08E7BE08-52D5-DE11-9635-001731AF6B83.root'
        )
)
process.DQMStore.referenceFileName = 'file:./DQM_V0001_R000000001__RelValLM1_sfts__CMSSW_3_4_2-MC_3XY_V15-v1__GEN-SIM-RECO.root'
process.dqmSaver.referenceHandling = 'all'


process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
)

process.p = cms.Path(process.genCandidatesForMET*process.genParticlesForMETAllVisible*process.genMetTrue*process.HLTSusyExoVal*process.SusyExoPostVal*process.hltSusyExoQualityTester)
process.pEnd = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
