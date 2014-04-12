import FWCore.ParameterSet.Config as cms


ttbarRECO = cms.untracked.vstring(
     '/store/mc/Spring10/TTbarJets-madgraph/AODSIM/START3X_V26_S09-v1/0005/0210B899-9C46-DF11-A10F-003048C69294.root'
    ,'/store/mc/Spring10/TTbarJets-madgraph/AODSIM/START3X_V26_S09-v1/0005/0C39D8AD-A846-DF11-8016-003048C692CA.root'
)

ttbarJets = cms.untracked.vstring(
     '/store/user/rwolf/school/patTuple_ttbar_jets.root'
)

zjetsRECO = cms.untracked.vstring(
     '/store/mc/Spring10/ZJets-madgraph/AODSIM/START3X_V26_S09-v1/0013/00EFC4EA-3847-DF11-A194-003048D4DF80.root'
    ,'/store/mc/Spring10/ZJets-madgraph/AODSIM/START3X_V26_S09-v1/0013/0C096217-3A47-DF11-9E65-003048C692A4.root'
)

zjetsTracks  = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_zjets_tracks.root'
)

zjetsTrigger  = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_zjets_trigger.root'
)

## 299,991 QCD events as defined on WorkBookPATExampleTopQuarks
simulationQCD = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_qcd_0.root'
   ,'/store/user/rwolf/school/patTuple_qcd_1.root'
   ,'/store/user/rwolf/school/patTuple_qcd_2.root'
   ,'/store/user/rwolf/school/patTuple_qcd_3.root'
   ,'/store/user/rwolf/school/patTuple_qcd_4.root'
   ,'/store/user/rwolf/school/patTuple_qcd_5.root'
   ,'/store/user/rwolf/school/patTuple_qcd_6.root'
   ,'/store/user/rwolf/school/patTuple_qcd_7.root'
   ,'/store/user/rwolf/school/patTuple_qcd_8.root'
   ,'/store/user/rwolf/school/patTuple_qcd_9.root'
   ,'/store/user/rwolf/school/patTuple_qcd_10.root'
   ,'/store/user/rwolf/school/patTuple_qcd_11.root'
   ,'/store/user/rwolf/school/patTuple_qcd_12.root'
   ,'/store/user/rwolf/school/patTuple_qcd_13.root'
   ,'/store/user/rwolf/school/patTuple_qcd_14.root'
   ,'/store/user/rwolf/school/patTuple_qcd_15.root'
   ,'/store/user/rwolf/school/patTuple_qcd_16.root'
   ,'/store/user/rwolf/school/patTuple_qcd_17.root'
   ,'/store/user/rwolf/school/patTuple_qcd_18.root'
   ,'/store/user/rwolf/school/patTuple_qcd_19.root'
   ,'/store/user/rwolf/school/patTuple_qcd_20.root'
   ,'/store/user/rwolf/school/patTuple_qcd_21.root'
   ,'/store/user/rwolf/school/patTuple_qcd_22.root'
   ,'/store/user/rwolf/school/patTuple_qcd_23.root'
   ,'/store/user/rwolf/school/patTuple_qcd_24.root'
   ,'/store/user/rwolf/school/patTuple_qcd_25.root'
   ,'/store/user/rwolf/school/patTuple_qcd_26.root'
   ,'/store/user/rwolf/school/patTuple_qcd_27.root'
   ,'/store/user/rwolf/school/patTuple_qcd_28.root'
   ,'/store/user/rwolf/school/patTuple_qcd_29.root'
)

##  99,991 W+Jets events as defined on WorkBookPATExampleTopQuarks
simulationWjets = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_wjets_0.root'
   ,'/store/user/rwolf/school/patTuple_wjets_1.root'
   ,'/store/user/rwolf/school/patTuple_wjets_2.root'
   ,'/store/user/rwolf/school/patTuple_wjets_3.root'
   ,'/store/user/rwolf/school/patTuple_wjets_4.root'
   ,'/store/user/rwolf/school/patTuple_wjets_5.root'
   ,'/store/user/rwolf/school/patTuple_wjets_6.root'
   ,'/store/user/rwolf/school/patTuple_wjets_7.root'
   ,'/store/user/rwolf/school/patTuple_wjets_8.root'
   ,'/store/user/rwolf/school/patTuple_wjets_9.root'
)

##   9,991 Z+Jets events as defined on WorkBookPATExampleTopQuarks
simulationZjets = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_zjets_0.root'
   ,'/store/user/rwolf/school/patTuple_zjets_1.root'
   ,'/store/user/rwolf/school/patTuple_zjets_2.root'
   ,'/store/user/rwolf/school/patTuple_zjets_3.root'
   ,'/store/user/rwolf/school/patTuple_zjets_4.root'
   ,'/store/user/rwolf/school/patTuple_zjets_5.root'
   ,'/store/user/rwolf/school/patTuple_zjets_6.root'
   ,'/store/user/rwolf/school/patTuple_zjets_7.root'
   ,'/store/user/rwolf/school/patTuple_zjets_8.root'
   ,'/store/user/rwolf/school/patTuple_zjets_9.root'
)

##   1000 Ttbar events as defined on WorkBookPATExampleTopQuarks
simulationTtbar = cms.untracked.vstring(
    '/store/user/rwolf/school/patTuple_ttbar.root'
)

##   ~14000000 events of the muon skim corresponding to 2/pb as
##   defined on WorkBookPATExampleTopQuarks
muonSKIM  = cms.untracked.vstring(
     '/store/user/rwolf/test/patTuple_10_1.root'
    ,'/store/user/rwolf/test/patTuple_11_1.root'
    ,'/store/user/rwolf/test/patTuple_12_2.root'
    ,'/store/user/rwolf/test/patTuple_13_1.root'
    ,'/store/user/rwolf/test/patTuple_14_2.root'
    ,'/store/user/rwolf/test/patTuple_15_1.root'
    ,'/store/user/rwolf/test/patTuple_16_2.root'
    ,'/store/user/rwolf/test/patTuple_17_1.root'
    ,'/store/user/rwolf/test/patTuple_17_2.root'
    ,'/store/user/rwolf/test/patTuple_18_2.root'
    ,'/store/user/rwolf/test/patTuple_19_2.root'
    ,'/store/user/rwolf/test/patTuple_1_1.root'
    ,'/store/user/rwolf/test/patTuple_1_2.root'
    ,'/store/user/rwolf/test/patTuple_20_1.root'
    ,'/store/user/rwolf/test/patTuple_21_1.root'
    ,'/store/user/rwolf/test/patTuple_24_2.root'
    ,'/store/user/rwolf/test/patTuple_25_1.root'
    ,'/store/user/rwolf/test/patTuple_26_1.root'
    ,'/store/user/rwolf/test/patTuple_27_1.root'
    ,'/store/user/rwolf/test/patTuple_28_2.root'
    ,'/store/user/rwolf/test/patTuple_29_1.root'
    ,'/store/user/rwolf/test/patTuple_30_2.root'
    ,'/store/user/rwolf/test/patTuple_31_1.root'
    ,'/store/user/rwolf/test/patTuple_32_1.root'
    ,'/store/user/rwolf/test/patTuple_34_2.root'
    ,'/store/user/rwolf/test/patTuple_35_2.root'
    ,'/store/user/rwolf/test/patTuple_36_2.root'
    ,'/store/user/rwolf/test/patTuple_37_2.root'
    ,'/store/user/rwolf/test/patTuple_39_2.root'
    ,'/store/user/rwolf/test/patTuple_40_2.root'
    ,'/store/user/rwolf/test/patTuple_41_2.root'
    ,'/store/user/rwolf/test/patTuple_42_2.root'
    ,'/store/user/rwolf/test/patTuple_44_2.root'
    ,'/store/user/rwolf/test/patTuple_45_1.root'
    ,'/store/user/rwolf/test/patTuple_46_1.root'
    ,'/store/user/rwolf/test/patTuple_47_1.root'
    ,'/store/user/rwolf/test/patTuple_48_1.root'
    ,'/store/user/rwolf/test/patTuple_49_2.root'
    ,'/store/user/rwolf/test/patTuple_50_1.root'
    ,'/store/user/rwolf/test/patTuple_51_1.root'
    ,'/store/user/rwolf/test/patTuple_52_1.root'
    ,'/store/user/rwolf/test/patTuple_53_1.root'
    ,'/store/user/rwolf/test/patTuple_54_1.root'
    ,'/store/user/rwolf/test/patTuple_55_1.root'
    ,'/store/user/rwolf/test/patTuple_56_1.root'
    ,'/store/user/rwolf/test/patTuple_57_1.root'
    ,'/store/user/rwolf/test/patTuple_58_2.root'
    ,'/store/user/rwolf/test/patTuple_59_2.root'
    ,'/store/user/rwolf/test/patTuple_5_1.root'
    ,'/store/user/rwolf/test/patTuple_5_2.root'
    ,'/store/user/rwolf/test/patTuple_60_1.root'
    ,'/store/user/rwolf/test/patTuple_61_1.root'
    ,'/store/user/rwolf/test/patTuple_62_1.root'
    ,'/store/user/rwolf/test/patTuple_63_2.root'
    ,'/store/user/rwolf/test/patTuple_64_2.root'
    ,'/store/user/rwolf/test/patTuple_65_1.root'
    ,'/store/user/rwolf/test/patTuple_66_2.root'
    ,'/store/user/rwolf/test/patTuple_67_1.root'
    ,'/store/user/rwolf/test/patTuple_68_1.root'
    ,'/store/user/rwolf/test/patTuple_69_1.root'
    ,'/store/user/rwolf/test/patTuple_6_1.root'
    ,'/store/user/rwolf/test/patTuple_70_1.root'
    ,'/store/user/rwolf/test/patTuple_71_1.root'
    ,'/store/user/rwolf/test/patTuple_72_2.root'
    ,'/store/user/rwolf/test/patTuple_73_2.root'
    ,'/store/user/rwolf/test/patTuple_74_1.root'
    ,'/store/user/rwolf/test/patTuple_75_1.root'
    ,'/store/user/rwolf/test/patTuple_77_2.root'
    ,'/store/user/rwolf/test/patTuple_79_2.root'
    ,'/store/user/rwolf/test/patTuple_7_1.root'
    ,'/store/user/rwolf/test/patTuple_80_2.root'
    ,'/store/user/rwolf/test/patTuple_81_1.root'
    ,'/store/user/rwolf/test/patTuple_8_1.root'
    ,'/store/user/rwolf/test/patTuple_9_1.root'
)

# CMSSW_3_8_6 re-reconstruction of muon PD, run 144112, 17717 events AOD
dataMu = cms.untracked.vstring(
    '/store/data/Run2010A/Mu/AOD/Nov4ReReco_v1/0011/D2E5D86F-AEEC-DF11-B261-0017A4771028.root'
)
