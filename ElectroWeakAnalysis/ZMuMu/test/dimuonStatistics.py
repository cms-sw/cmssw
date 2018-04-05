import FWCore.ParameterSet.Config as cms

process = cms.Process("dimuonStatistics")

process.include("FWCore/MessageLogger/data/MessageLogger.cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
  "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_1.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_10.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_11.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_12.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_13.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_14.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_15.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_16.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_17.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_18.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_19.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_2.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_20.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_21.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_22.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_23.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_24.root",
 "rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_25.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_26.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_27.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_28.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_29.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_3.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_30.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_31.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_32.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_33.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_34.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_35.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_36.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_37.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_38.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_39.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_4.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_40.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_41.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_42.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_43.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_44.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_45.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_46.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_47.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_48.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_49.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_5.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_50.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_51.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_52.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_53.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_54.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_55.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_56.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_57.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_58.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_59.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_6.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_60.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_61.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_62.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_63.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_64.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_65.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_66.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_67.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_68.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_69.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_7.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_70.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_71.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_72.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_73.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_74.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_75.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_76.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_77.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_78.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_79.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_8.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_80.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_81.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_82.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_83.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_84.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_85.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_86.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_87.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_88.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_89.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_9.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_90.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_91.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_92.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_93.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_94.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_95.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_96.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_97.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_98.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_99.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_100.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_101.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_102.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_103.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_104.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_105.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_106.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_107.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_108.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_109.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_110.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_111.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_112.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_113.root",
"rfio:/dpm/na.infn.it/home/cms/store/user/degrutto/EWK_ZMM_OCT_EX/zmm_v2/testZMuMuSubSkim_114.root",
   )
)
zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.0 & abs(daughter(1).eta)<2.0 & mass > 20"),
    isoCut = cms.double(3.),
    ptThreshold = cms.untracked.double(1.5),
    etEcalThreshold = cms.untracked.double(0.2),
    etHcalThreshold = cms.untracked.double(0.5),
    deltaRVetoTrk = cms.untracked.double(0.015),
    deltaRTrk = cms.untracked.double(0.3),
    deltaREcal = cms.untracked.double(0.25),
    deltaRHcal = cms.untracked.double(0.25),
    alpha = cms.untracked.double(0.),
    beta = cms.untracked.double(-0.75),
    relativeIsolation = cms.bool(False)

# For standard isolation (I_Tkr<3GeV) choose this configuration:
#   isoCut = cms.double(3.),
#   ptThreshold = cms.untracked.double(1.5),
#   etEcalThreshold = cms.untracked.double(0.2),
#   etHcalThreshold = cms.untracked.double(0.5),
#   deltaRVetoTrk = cms.untracked.double(0.015),
#   deltaRTrk = cms.untracked.double(0.3),
#   deltaREcal = cms.untracked.double(0.25),
#   deltaRHcal = cms.untracked.double(0.25),
#   alpha = cms.untracked.double(0.),
#   beta = cms.untracked.double(-0.75),
#   relativeIsolation = cms.bool(False)


 )

process.goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)
#ZMuMu: richiedo almeno 1 HLT trigger match.Per la shape
process.goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

#ZMuMu: richiedo 2 HLT trigger match
process.goodZToMuMu2HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("bothMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

#ZMuMu: richiedo 1 HLT trigger match
process.goodZToMuMu1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("exactlyOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


process.nonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

#ZMuMu1notIso: richiedo almeno un trigger
process.nonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("nonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

process.zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneTrack = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    filter = cms.bool(True)
)

process.zToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    filter = cms.bool(True)
)

process.goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

#ZMuTk:richiedo che il muGlobal 'First' ha HLT match
process.goodZToMuMuOneTrackFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrack"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

process.goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuOneStandAloneMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

#ZMuSta:richiedo che il muGlobal ha HLT match
process.goodZToMuMuOneStandAloneMuonFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


goodZToMuMuTemplate = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("replace this string with your cut"),
    src = cms.InputTag("goodZToMuMuAtLeast1HLT"),
    filter = cms.bool(False)
)




process.DimuGlobalNotIsoStat = cms.EDAnalyzer(
    "DimuonStatistics",
    src = cms.InputTag("dimuonsGlobal"), # dimuonsOneTrack, dimuonsOneStandAlone
    ptMin = cms.untracked.double(20.0),
    massMin = cms.untracked.double(60.0),
    massMax = cms.untracked.double(120.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(2.0),
    trkIso = cms.untracked.double(10000)
    )

process.DimuGlobalIsoStat = cms.EDAnalyzer(
    "DimuonStatistics",
    src = cms.InputTag("dimuonsGlobal"), 
    ptMin = cms.untracked.double(20.0),
    massMin = cms.untracked.double(60.0),
    massMax = cms.untracked.double(120.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(2.0),
    trkIso = cms.untracked.double(3.0)
    )

process.DimuOneTrackIsoStat=cms.EDAnalyzer(
    "DimuonStatistics",
    src = cms.InputTag("dimuonsOneTrack"), 
    ptMin = cms.untracked.double(20.0),
    massMin = cms.untracked.double(60.0),
    massMax = cms.untracked.double(120.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(2.0),
    trkIso = cms.untracked.double(3.0)
    )
process.DimuOneStaIsoStat=cms.EDAnalyzer(
    "DimuonStatistics",
    src = cms.InputTag("dimuonsOneStandAloneMuon"), 
    ptMin = cms.untracked.double(20.0),
    massMin = cms.untracked.double(60.0),
    massMax = cms.untracked.double(120.0),
    etaMin = cms.untracked.double(-1.0),
    etaMax = cms.untracked.double(2.0),
    trkIso = cms.untracked.double(3.0)
    )
 



process.OneHLTIsolatedPath = cms.Path(
    process.goodZToMuMu +
    process.goodZToMuMu1HLT +
    process.DimuGlobalIsoStat
     )

process.TwoHLTIsolatedPath = cms.Path(
    process.goodZToMuMu +
    process.goodZToMuMu2HLT +
    process.DimuGlobalIsoStat
     )

process.NonIsolatedPath = cms.Path(
   process.nonIsolatedZToMuMu *
   process.nonIsolatedZToMuMuAtLeast1HLT*
   process.DimuGlobalNotIsoStat
    )


process.MuStaIsolatedPath=cms.Path(
    ~process.goodZToMuMu + 
    process.zToMuMuOneStandAloneMuon + 
    process.goodZToMuMuOneStandAloneMuon +
    process.goodZToMuMuOneStandAloneMuonFirstHLT +
    process.DimuOneStaIsoStat
  )
    
process.MuTkIsolatedPath = cms.Path(
    ~process.goodZToMuMu + 
    ~process.zToMuMuOneStandAloneMuon +
    process.zToMuGlobalMuOneTrack +
    process.zToMuMuOneTrack +
    process.goodZToMuMuOneTrack +
    process.goodZToMuMuOneTrackFirstHLT +
    process.DimuOneTrackIsoStat

    )




