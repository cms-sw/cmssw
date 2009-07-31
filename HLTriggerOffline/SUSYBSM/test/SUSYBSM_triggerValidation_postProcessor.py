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

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_3_1_0/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V1-v1/0002/2A2BB3C2-DE66-DE11-8A25-001D09F27067.root',
    #'/store/relval/CMSSW_3_1_0/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V1-v1/0001/D62DED56-8F66-DE11-96AF-001D09F241B4.root',
    #'/store/relval/CMSSW_3_1_0/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V1-v1/0001/9E488F5D-9066-DE11-85E9-001D09F2543D.root',
    #'/store/relval/CMSSW_3_1_0/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V1-v1/0001/58B73B6D-9266-DE11-AFC2-001617DBCF90.root',
    #'/store/relval/CMSSW_3_1_0/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V1-v1/0001/28ACB7E0-9266-DE11-AFCF-001D09F25325.root',
    #'/store/relval/CMSSW_3_1_0/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V1-v1/0001/02841A84-9966-DE11-A26B-001D09F23A61.root',
    '/store/relval/CMSSW_3_1_0/RelValLM1_sfts/GEN-SIM-RECO/MC_31X_V1-v1/0001/024EE1A1-8166-DE11-965F-001D09F2543D.root'
#"file:/build/nuno/test31/CMSSW_3_1_0_pre5/src/TTbar_Tauola_cfi_py_GEN_FASTSIM_VALIDATION.root"
#"file:myreco2_RAW2DIGI_RECO_ALCA_VALIDATION.root"
#'/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0000/E63C1A00-0C2C-DE11-BFC1-000423D98800.root'
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
