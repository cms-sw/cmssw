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
        '/store/relval/CMSSW_3_6_0/RelValLM1_sfts/GEN-SIM-RECO/MC_36Y_V4-v1/0014/D8F19D3E-AC49-DF11-9F9A-002618943852.root',
        '/store/relval/CMSSW_3_6_0/RelValLM1_sfts/GEN-SIM-RECO/MC_36Y_V4-v1/0014/D60BAD49-FC49-DF11-9A39-003048678E8A.root',
        '/store/relval/CMSSW_3_6_0/RelValLM1_sfts/GEN-SIM-RECO/MC_36Y_V4-v1/0014/C8526DD3-AB49-DF11-888E-001A92971ADC.root',
        '/store/relval/CMSSW_3_6_0/RelValLM1_sfts/GEN-SIM-RECO/MC_36Y_V4-v1/0014/86314B81-AB49-DF11-B855-00304867BEDE.root',
        '/store/relval/CMSSW_3_6_0/RelValLM1_sfts/GEN-SIM-RECO/MC_36Y_V4-v1/0014/7290CE76-AB49-DF11-AD07-00261894388A.root',
        '/store/relval/CMSSW_3_6_0/RelValLM1_sfts/GEN-SIM-RECO/MC_36Y_V4-v1/0014/1E2E323C-AC49-DF11-A632-002354EF3BDA.root',
        '/store/relval/CMSSW_3_6_0/RelValLM1_sfts/GEN-SIM-RECO/MC_36Y_V4-v1/0013/EED059EC-9849-DF11-B324-001A92971B06.root'


        )
)
process.DQMStore.referenceFileName = 'file:./DQM_V0001_R000000001__RelValLM1_sfts__CMSSW_3_6_0_pre1-MC_3XY_V21-v2__GEN-SIM-RECO.root'
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
