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

process.load("HLTriggerOffline.SUSYBSM.SUSYBSM_triggerValidation_cff")
process.load("HLTriggerOffline.SUSYBSM.SUSYBSM_postProcessor_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_0_0_pre7/RelValLM1_sfts/GEN-SIM-RECO/IDEAL_30X_v1/0006/0C5E2705-6AE8-DD11-95EF-0030487A3C9A.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValLM1_sfts/GEN-SIM-RECO/IDEAL_30X_v1/0006/40E30846-70E8-DD11-A5F3-000423D944F0.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValLM1_sfts/GEN-SIM-RECO/IDEAL_30X_v1/0006/6C9170F8-71E8-DD11-9B50-001D09F241B9.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValLM1_sfts/GEN-SIM-RECO/IDEAL_30X_v1/0006/BC47E3BD-6AE8-DD11-8B0D-001D09F2932B.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValLM1_sfts/GEN-SIM-RECO/IDEAL_30X_v1/0006/CE8ABD28-6CE8-DD11-9A94-000423D94990.root'
        )
)

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

process.p = cms.Path(process.genCandidatesForMET*process.genParticlesForMETAllVisible*process.genMetTrue*process.HLTSusyExoVal)
process.pEnd = cms.EndPath(process.SusyExoPostVal+process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
