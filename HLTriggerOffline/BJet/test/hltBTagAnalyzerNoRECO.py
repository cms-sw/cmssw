import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTValidation")

# TFileService
process.load("PhysicsTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = 'plots.root'

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append('HLTBtagAnalyzer')

process.load("HLTriggerOffline.BJet.hltJetMCTools_cff")
#process.load("HLTriggerOffline.BJet.hltBLifetimeExtra_cff")
process.load("HLTriggerOffline.BJet.hltBLifetime_cff")
process.load("HLTriggerOffline.BJet.hltBLifetimeRelaxed_cff")
#process.load("HLTriggerOffline.BJet.hltBSoftmuonExtra_cff")
process.load("HLTriggerOffline.BJet.hltBSoftmuon_cff")
process.load("HLTriggerOffline.BJet.hltBSoftmuonRelaxed_cff")
process.MessageLogger.debugModules.extend(process.hltBLifetime_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBLifetimeRelaxed_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBSoftmuon_modules.modules)
process.MessageLogger.debugModules.extend(process.hltBSoftmuonRelaxed_modules.modules)

# don't use MC matching
process.hlt_BTagIP_Jet120_Relaxed.mcPartons      = 'none'
process.hlt_BTagIP_Jet180.mcPartons              = 'none'
process.hlt_BTagIP_DoubleJet120.mcPartons        = 'none'
process.hlt_BTagIP_DoubleJet60_Relaxed.mcPartons = 'none'
process.hlt_BTagIP_TripleJet40_Relaxed.mcPartons = 'none'
process.hlt_BTagIP_TripleJet70.mcPartons         = 'none'
process.hlt_BTagIP_QuadJet30_Relaxed.mcPartons   = 'none'
process.hlt_BTagIP_QuadJet40.mcPartons           = 'none'
process.hlt_BTagIP_HT320_Relaxed.mcPartons       = 'none'
process.hlt_BTagIP_HT470.mcPartons               = 'none'
process.hlt_BTagMu_DoubleJet120.mcPartons        = 'none'
process.hlt_BTagMu_DoubleJet60_Relaxed.mcPartons = 'none'
process.hlt_BTagMu_TripleJet40_Relaxed.mcPartons = 'none'
process.hlt_BTagMu_TripleJet70.mcPartons         = 'none'
process.hlt_BTagMu_QuadJet30_Relaxed.mcPartons   = 'none'
process.hlt_BTagMu_QuadJet40.mcPartons           = 'none'
process.hlt_BTagMu_HT250_Relaxed.mcPartons       = 'none'
process.hlt_BTagMu_HT370.mcPartons               = 'none'
process.hlt_BTagMu_Jet20_Calib.mcPartons         = 'none'

# don't use RECO matching
process.hlt_BTagIP_Jet120_Relaxed.offlineBJets      = 'none'
process.hlt_BTagIP_Jet180.offlineBJets              = 'none'
process.hlt_BTagIP_DoubleJet120.offlineBJets        = 'none'
process.hlt_BTagIP_DoubleJet60_Relaxed.offlineBJets = 'none'
process.hlt_BTagIP_TripleJet40_Relaxed.offlineBJets = 'none'
process.hlt_BTagIP_TripleJet70.offlineBJets         = 'none'
process.hlt_BTagIP_QuadJet30_Relaxed.offlineBJets   = 'none'
process.hlt_BTagIP_QuadJet40.offlineBJets           = 'none'
process.hlt_BTagIP_HT320_Relaxed.offlineBJets       = 'none'
process.hlt_BTagIP_HT470.offlineBJets               = 'none'
process.hlt_BTagMu_DoubleJet120.offlineBJets        = 'none'
process.hlt_BTagMu_DoubleJet60_Relaxed.offlineBJets = 'none'
process.hlt_BTagMu_TripleJet40_Relaxed.offlineBJets = 'none'
process.hlt_BTagMu_TripleJet70.offlineBJets         = 'none'
process.hlt_BTagMu_QuadJet30_Relaxed.offlineBJets   = 'none'
process.hlt_BTagMu_QuadJet40.offlineBJets           = 'none'
process.hlt_BTagMu_HT250_Relaxed.offlineBJets       = 'none'
process.hlt_BTagMu_HT370.offlineBJets               = 'none'
process.hlt_BTagMu_Jet20_Calib.offlineBJets         = 'none'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/244B5962-6BE9-DD11-A031-000423D6BA18.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/264114F5-6BE9-DD11-8A72-000423D98844.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/2ABF3DD4-77E9-DD11-9172-001617C3B710.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/2CF4D410-6DE9-DD11-A8B5-000423D9870C.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/30031C00-79E9-DD11-99B7-001D09F290BF.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/3A4759AD-77E9-DD11-A679-000423D6B5C4.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/42026D38-89E9-DD11-8662-000423D94E70.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/4268DF31-67E9-DD11-896B-000423D94494.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/56241D73-6DE9-DD11-9F1E-00304879FA4A.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/5CEACBF9-8EE9-DD11-94B8-000423D99EEE.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/72C24A21-73E9-DD11-A7CA-001D09F23944.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/72F21BF8-6DE9-DD11-BC43-000423D8FA38.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/74A85116-80E9-DD11-9CEB-00304879FBB2.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/9294EED4-74E9-DD11-A1A2-001D09F232B9.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/980CCB9F-7CE9-DD11-B1A4-001D09F251CC.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/9E34304D-69E9-DD11-AE0B-001617C3B6E2.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/C08E8D40-6AE9-DD11-BA20-001D09F28755.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/DC6101D6-71E9-DD11-882C-000423D98FBC.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/DCC52C53-7EE9-DD11-94E1-000423D951D4.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/E2E0D0A5-70E9-DD11-AD32-001D09F25442.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/EC9D3AF6-76E9-DD11-AADF-001617C3B710.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/F2A38876-74E9-DD11-916C-001D09F24600.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/F88F071A-65E9-DD11-A133-000423D99658.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/FAF21164-3CEA-DD11-813D-000423D951D4.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_30X_LowLumiPileUp_v1/0006/FE76BFFA-84E9-DD11-87DD-000423D98C20.root'
    )
)

#process.extra_lifetime    = cms.Path( process.hltBLifetimeExtra )
#process.extra_lifetime_rx = cms.Path( process.hltBLifetimeExtraRelaxed )
#process.extra_softmuon    = cms.Path( process.hltBSoftmuonExtra )
#process.extra_softmuon_rx = cms.Path( process.hltBSoftmuonExtraRelaxed )
#process.extra_softmuon_dr = cms.Path( process.hltBSoftmuonExtraByDR )
#process.extra_jetmctools  = cms.Path( process.hltJetMCTools )
process.validation        = cms.Path( process.hltBLifetime + process.hltBLifetimeRelaxed + process.hltBSoftmuon + process.hltBSoftmuonRelaxed )

process.schedule = cms.Schedule(
#    process.extra_lifetime,
#    process.extra_lifetime_rx,
#    process.extra_softmuon,
#    process.extra_softmuon_rx,
#    process.extra_softmuon_dr,
#    process.extra_jetmctools,
    process.validation)
