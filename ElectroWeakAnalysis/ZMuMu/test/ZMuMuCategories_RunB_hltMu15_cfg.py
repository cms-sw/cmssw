import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkZMuMuCategories")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
#process.GlobalTag.globaltag = cms.string('START3X_V26::All') 
process.GlobalTag.globaltag = cms.string('START38_V13::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_100_1_uSp.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_101_1_xWO.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_102_1_BTF.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_103_1_NMT.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_104_1_Fef.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_105_1_HZB.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_106_1_RE4.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_107_1_TDZ.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_108_1_8kb.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_109_1_qxw.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_10_1_ks4.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_110_1_pzz.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_111_1_bRx.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_112_1_gv3.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_113_1_No8.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_114_1_hZV.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_115_1_U8I.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_116_1_U4R.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_117_1_TJZ.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_118_1_pbB.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_119_1_DGd.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_11_1_pOb.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_120_1_Ae4.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_12_1_w6i.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_13_1_ohA.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_14_1_nFD.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_15_1_Plf.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_16_1_yqQ.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_17_1_uEl.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_18_1_Vk6.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_19_1_N1I.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_1_1_AHg.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_20_1_C44.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_21_1_UvT.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_22_1_0ZH.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_23_1_vfy.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_24_1_rXO.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_25_1_cl1.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_26_1_tKG.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_27_1_qT6.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_28_1_i6S.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_29_1_JZn.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_2_1_rSt.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_30_1_xBS.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_31_1_gTb.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_32_1_zhE.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_33_1_BYY.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_34_1_Iup.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_35_1_3lr.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_36_1_7Bj.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_37_1_Z5A.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_38_1_4jD.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_39_1_dcn.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_3_1_AYf.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_40_1_HNb.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_41_1_TsM.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_42_1_QZ5.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_43_1_dNL.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_44_1_msU.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_45_1_MED.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_46_1_dfO.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_47_1_xU2.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_48_1_TrJ.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_49_1_Vv3.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_4_1_swg.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_50_1_2Oq.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_51_1_6d9.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_52_1_t5D.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_53_1_tTK.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_54_1_zbo.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_55_1_RSL.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_56_1_Hb1.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_57_1_Qyi.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_58_1_JYj.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_59_1_rW3.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_5_1_82A.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_60_1_xQT.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_61_1_4MO.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_62_1_njm.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_63_1_cgX.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_64_1_vKR.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_65_1_UbW.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_66_1_BYC.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_67_1_gzU.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_68_1_YLv.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_69_1_nqt.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_6_1_Eyk.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_70_1_5Re.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_71_1_qg0.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_72_1_p1c.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_73_1_tf9.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_74_1_1Bf.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_75_1_VzI.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_76_1_6Yi.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_77_1_q17.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_78_1_0Mr.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_79_1_k79.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_7_1_EL5.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_80_1_Pn0.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_81_1_Ufo.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_82_1_KPe.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_83_1_K58.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_84_1_axW.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_85_1_pfo.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_86_1_rMM.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_87_1_Aca.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_88_1_L91.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_89_1_Xav.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_8_1_KhE.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_90_1_i8R.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_91_1_LnA.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_92_1_UI8.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_93_1_EMj.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_94_1_msK.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_95_1_YfA.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_96_1_HHk.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_97_1_AQx.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_98_1_oxL.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_99_1_liF.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degruttola//2010/HLTMu15_run2010B_Dec22ReReco/degrutto/Mu/ZMuMuSubSkim_Run2010B_hltMu15-rereco_4nov/3c0b86d62ce733c2557fd62cee7ed9f6//testZMuMuSubskim_9_1_uaS.root",

    )
)

# replace ZSelection if wanted......
## from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import * 
## zSelection.cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 0")

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff")

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ewkZMuMuCategories.root")
)


### vertexing
#process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesVtxed_cff")

### plots

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesPlots_cff")

### ntuple

### Added UserData

#process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesNtuples_cff")
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisNtupler_cff")
process.ntuplesOut.fileName = cms.untracked.string('file:Ntuple_39X_RunB_hltmu15.root')

############# change the hlt default path..........
process.goodZToMuMuAtLeast1HLT.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuAtLeast1HLTLoose.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuAB1HLTLoose.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuAB1HLT.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuOneStandAloneMuonFirstHLTLoose.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuOneStandAloneMuonFirstHLT.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuOneTrackFirstHLTLoose.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuOneTrackFirstHLT.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuOneTrackerMuonFirstHLTLoose.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuOneTrackerMuonFirstHLT.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuSameChargeAtLeast1HLTLoose.hltPath= cms.string("HLT_Mu15_v1")
process.goodZToMuMuSameChargeAtLeast1HLT.hltPath= cms.string("HLT_Mu15_v1")
process.nonIsolatedZToMuMuAtLeast1HLT.hltPath= cms.string("HLT_Mu15_v1")
process.oneNonIsolatedZToMuMuAtLeast1HLT.hltPath= cms.string("HLT_Mu15_v1")
process.twoNonIsolatedZToMuMuAtLeast1HLT.hltPath= cms.string("HLT_Mu15_v1")
