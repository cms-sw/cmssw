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
#        'file:/tmp/chiorbo/EC43AC0A-3305-DF11-99FE-0030487CD13A.root'
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/EEDE7700-C50F-DF11-A8EC-00304867342C.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/8E8BEF2D-4C0F-DF11-AD90-0030487D0D3A.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/7A70652A-4F0F-DF11-A4F2-0030487CD704.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/764459F0-4C0F-DF11-9A0E-00304879BAB2.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/4CDBE375-430F-DF11-88BE-0030487C635A.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/3C57A952-4E0F-DF11-B002-0030487CD704.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/36E9495E-4D0F-DF11-8238-00304879EDEA.root',
        '/store/relval/CMSSW_3_5_0_pre5/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V20-v2/0009/0ED771FF-4A0F-DF11-AE23-0030487A195C.root'

#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/EC43AC0A-3305-DF11-99FE-0030487CD13A.root',
#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/D8630372-3105-DF11-87A3-0030487CD13A.root',
#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/BECC9492-3005-DF11-9CBD-0030487A1884.root',
#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/76616E1A-B405-DF11-A263-0030487A1990.root',
#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/682B987D-3905-DF11-B952-0030487A17B8.root',
#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/3C144158-3005-DF11-9933-0030487CD906.root',
#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/3A72A401-3105-DF11-A1D6-0030487CD906.root',
#        '/store/relval/CMSSW_3_5_0_pre3/RelValLM1_sfts/GEN-SIM-RECO/MC_3XY_V15-v2/0006/26DBA1F2-2E05-DF11-B620-0030487A322E.root'
        )
)
#process.DQMStore.referenceFileName = 'file:./330pre3/DQM_V0001_LM1_330pre3_R000000001.root'
#process.dqmSaver.referenceHandling = 'all'


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
