import FWCore.ParameterSet.Config as cms

process = cms.Process("CHECKEFF")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# from path HLT_DoubleIsoEle10_L1I
process.load("HLTriggerOffline.Egamma.doubleElectronCE_cfi")

# from path HLT_DoubleIsoEle12_L1R
process.load("HLTriggerOffline.Egamma.doubleElectronRelaxedCE_cfi")

# from path HLT_IsoEle15_L1I
process.load("HLTriggerOffline.Egamma.singleElectronCE_cfi")

# not available 
# process.load("HLTriggerOffline.Egamma.singleElectronStartUpCE_cfi")

# from path HLT_IsoEle15_LW_L1I
process.load("HLTriggerOffline.Egamma.singleElectronLargeWindowCE_cfi")

# from path HLT_IsoEle18_L1R
process.load("HLTriggerOffline.Egamma.singleElectronRelaxedCE_cfi")

# from path HLT_Ele15_SW_L1R
process.load("HLTriggerOffline.Egamma.singleElectronRelaxedStartUpCE_cfi")

# from path HLT_Ele15_LW_L1R
process.load("HLTriggerOffline.Egamma.singleElectronRelaxedLargeWindowCE_cfi")

process.TimerService = cms.Service("TimerService",
    useCPUtime = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('../test/rootfiles/efftest_Wenu_testMyFix.root')
)

## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_1.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_2.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_3.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_4.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_5.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_6.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_7.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testNewSeeds_8.root'
## )
## )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_1.root',
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_2.root',
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_3.root',
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_4.root',
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_5.root',
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_6.root',
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_7.root',
'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMyFix_8.root'
)
)

## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_1.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_2.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_3.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_4.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_5.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_6.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_7.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testClaudeFix_8.root'
## )
## )

## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_1.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_2.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_3.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_4.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_5.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_6.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_7.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeeds_8.root'
## )
## )

## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_1.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_2.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_3.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_4.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_5.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_6.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_7.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsJRV_8.root'
## )
## )

## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_1.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_2.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_3.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_4.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_5.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_6.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_7.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testMixedSeedsTID_8.root'
## )
## )

## process.source = cms.Source("PoolSource",
##    fileNames = cms.untracked.vstring(
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_1.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_2.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_3.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_4.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_5.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_6.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_7.root',
## 'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphiLW_8.root'
## )
## )

## process.source = cms.Source("PoolSource",
##    fileNames = cms.untracked.vstring(
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_1.root',
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_2.root',
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_3.root',
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_4.root',
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_5.root',
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_6.root',
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_7.root',
##'rfio:/castor/cern.ch/user/c/covarell/egamma-trig/NEWRECO+HLT/Wenu_217_HLT_testDetaDphi_8.root'
##)
##)

## process.source = cms.Source("PoolSource",
##     fileNames = cms.untracked.vstring(
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/0CCEC628-C57D-DD11-AEA0-001617E30D06.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/167202E0-C47D-DD11-8E76-000423D6006E.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/1E5C03E3-C47D-DD11-A04C-000423D987FC.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/20D6FFC9-C47D-DD11-A6BB-001617C3B77C.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/2C9F7A15-C57D-DD11-915E-000423D98920.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/40E21ACE-C47D-DD11-A596-001617C3B73A.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/4C660050-C37D-DD11-9A77-001617E30D40.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/58529018-C37D-DD11-8D5F-001617C3B6C6.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/66DE9EAA-C47D-DD11-932F-001617C3B706.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/7A512B83-C47D-DD11-ABAA-000423D6CA42.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/A2A1CF24-C57D-DD11-9182-000423D6006E.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/C868770A-C37D-DD11-936A-001617C3B710.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/CE47CE0F-C37D-DD11-A4DF-001617C3B6E2.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/D679C27B-C47D-DD11-A6CC-001617DF785A.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/DA5B8C8A-C47D-DD11-A35D-001617E30D06.root',
## '/store/relval/CMSSW_2_1_7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0001/F846B7F1-C47D-DD11-889D-000423D6BA18.root'
## )
## )

process.test = cms.Path(process.singleElectronCE+process.singleElectronLargeWindowCE+process.singleElectronRelaxedCE+process.singleElectronRelaxedLargeWindowCE+process.singleElectronRelaxedStartUpCE)

