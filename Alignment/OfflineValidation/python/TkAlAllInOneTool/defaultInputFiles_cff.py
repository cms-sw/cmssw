import FWCore.ParameterSet.Config as cms

filesDefaultMC_IsoMuon = cms.untracked.vstring(
    '/store/relval/CMSSW_20_0_0_pre1/RelValQCD_Pt_1800_2400_14/ALCARECO/TkAlMuonIsolated-150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/4b892577-915a-490e-9f66-2269648bf20d.root'
)

filesDefaultMC_DoubleMuon = cms.untracked.vstring(
    '/store/relval/CMSSW_12_5_0_pre5/RelValZMM_14/GEN-SIM-RECO/125X_mcRun3_2022_realistic_v3-v1/10000/068634c7-e8e2-4e46-a68e-97ebefed4868.root'
)

filesDefaultMC_DoubleElectron_string = '/store/relval/CMSSW_20_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/028e09cc-784b-41ab-a712-511d5bb67724.root'

filesDefaultMC_DoubleElectron = cms.untracked.vstring(filesDefaultMC_DoubleElectron_string)

filesDefaultMC_DoubleMuon_string = '/store/relval/CMSSW_20_0_0_pre1/RelValZMM_14/GEN-SIM-RECO/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU-v1/2590000/42c2b496-5687-4872-803f-f0daa1c6b2b9.root'

filesDefaultMC_DoubleMuonAlCa_string = '/store/relval/CMSSW_20_0_0_pre1/RelValZMM_14/ALCARECO/TkAlZMuMu-150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/d42b4f5d-7e50-461a-924c-9e95ad81194b.root'

filesDefaultMC_TTBarPU = cms.untracked.vstring(
    '/store/relval/CMSSW_12_5_0_pre5/RelValTTbar_14TeV/GEN-SIM-RECO/PU_125X_mcRun3_2022_realistic_v3-v1/10000/0136c33f-3ff9-4602-8578-906ae6e0160b.root'
)

filesDefaultMC_MinBiasPUPhase2 = cms.untracked.vstring(
    '/store/relval/CMSSW_12_5_3/RelValMinBias_14TeV/ALCARECO/TkAlMinBias-125X_mcRun4_realistic_v5_2026D88PU-v1/2590000/27b7ab93-1d2b-4f4a-a98e-68386c314b5e.root',
)

filesDefaultMC_DoubleMuonPUPhase_string = '/store/mc/Phase2Spring24DIGIRECOMiniAOD/DYJetsToMuMu_M-50_TuneCP5_14TeV-madgraphMLM-pythia8/ALCARECO/TkAlZMuMu-noPUALCA_TkAlnoPU_140X_mcRun4_realistic_v6_ext1-v2/110000/1022ba4a-65f7-4409-9069-f35247a7a8e3.root'

filesDefaultMC_MinBiasPUPhase2RECO = cms.untracked.vstring(
    '/store/relval/CMSSW_14_1_0_pre6/RelValMinBias_14TeV/GEN-SIM-RECO/PU_141X_mcRun4_realistic_v1_STD_2026D110_PU-v3/2560000/c22f1cbd-50e3-458e-aba9-b0a327e4c971.root'
)

filesDefaultMC_TTbarPhase2RECO = cms.untracked.vstring(
    '/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/1bf48ba4-bd8e-405e-845d-cfdfbfbed923.root',
)

filesDefaultData_JetHTRun2018D = cms.untracked.vstring(
    '/store/data/Run2018D/JetHT/ALCARECO/TkAlMinBias-12Nov2019_UL2018-v3/270000/AF80DFBE-C277-1142-AE8F-71FE4444370A.root'
)

filesDefaultData_JetHTRun2022A = cms.untracked.vstring(
    '/store/data/Run2022A/JetHT/ALCARECO/TkAlMinBias-PromptReco-v1/000/352/900/00000/96120499-f83f-4b90-a828-58d4c2d26350.root'
)

filesDefaultData_JetHTRun2018DHcalIsoTrk = cms.untracked.vstring(
    '/store/data/Run2018D/JetHT/ALCARECO/HcalCalIsoTrkFilter-12Nov2019_UL2018-v3/100000/075A1F1E-20B4-134D-9794-AD764DA6730D.root')

filesDefaultMC_NoPU = cms.untracked.vstring(
    '/store/relval/CMSSW_20_0_0_pre1/RelValMinBias_14TeV/GEN-SIM-RECO/150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/e084e752-0b18-4f3b-b910-5eadc930ed59.root'
)

filesDefaultMC_NoPU_AlCa = cms.untracked.vstring('/store/relval/CMSSW_20_0_0_pre1/RelValQCD_FlatPt_15_3000HS_14/ALCARECO/TkAlMinBias-150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/76b13a13-8751-43f8-8c1b-475dfb7f9178.root')

filesDefaultMC_Ideal900GeV = cms.untracked.vstring(
    '/store/relval/CMSSW_12_0_2_patch1/RelValMinBias_14TeV/ALCARECO/TkAlMinBias-120X_mcRun3_2021_realistic_forpp900GeV_IdealTkAlign_v1_IdealTkAlignmentJIRA137-v1/2580000/044c1fa8-9bdf-4c2e-9aa4-00ba966426a2.root'
)

filesDefaultMC_Realistic2022 = cms.untracked.vstring(
    '/store/relval/CMSSW_12_5_0_pre2/RelValMinBias_14TeV/GEN-SIM-RECO/124X_mcRun3_2022_realistic_v3-v1/2580000/c828c73c-32d1-45be-a8ae-b8af4b8f4952.root'
)

filesDefaultMC_Realistic2022_string = '/store/relval/CMSSW_12_5_0_pre2/RelValMinBias_14TeV/GEN-SIM-RECO/124X_mcRun3_2022_realistic_v3-v1/2580000/c828c73c-32d1-45be-a8ae-b8af4b8f4952.root'

filesDefaultData_Comissioning2022_Cosmics_string = '/store/data/Commissioning2022/Cosmics/ALCARECO/TkAlCosmics0T-PromptReco-v1/000/348/776/00000/96538f53-2088-422c-91a5-841d735a81a8.root'

filesDefaultData_MinBias2018B = cms.untracked.vstring(
    '/store/express/Run2018B/StreamExpress/ALCARECO/TkAlMinBias-Express-v1/000/317/212/00000/00F0EFA7-8D64-E811-A594-FA163EFC96CC.root'
)

filesDefaultData_HLTPhys2024I = cms.untracked.vstring(
    '/store/data/Run2024I/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v2/000/386/803/00000/a95259f9-a333-44f9-b30f-803642e97590.root'
)

filesDefaultData_Cosmics_string = "/store/data/Run2022G/Cosmics/ALCARECO/TkAlCosmics0T-PromptReco-v1/000/362/440/00000/47f31eaa-1c00-4f39-902b-a09fa19c27f2.root"
