import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkZMuMuCategories")



process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.GlobalTag.globaltag = cms.string('START3X_V18::All') 
process.load("Configuration.StandardSequences.MagneticField_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_1.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_2.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_3.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_4.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_5.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_6.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_7.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_8.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_9.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_10.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_11.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_12.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_13.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_14.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_15.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_16.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_17.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_18.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_19.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_20.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_21.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_22.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_23.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_24.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_25.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_26.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_27.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_28.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_29.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_30.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_31.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_32.root", 
# "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_33.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_34.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_35.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_36.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_37.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_38.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_39.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_40.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_41.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_42.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_43.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_44.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_45.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_46.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_47.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_48.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_49.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_50.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_51.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_52.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_53.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_54.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_55.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_56.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_57.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_58.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_59.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_60.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_61.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_62.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_63.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_64.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_65.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_66.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_67.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_68.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_69.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_70.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_71.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_72.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_73.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_74.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_75.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_76.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_77.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_78.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_79.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_80.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_81.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_82.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_83.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_84.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_85.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_86.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_87.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_88.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_89.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_90.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_91.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_92.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_93.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_94.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_95.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_96.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_97.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_98.root"
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_99.root", 
 "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_100.root",
  "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_101.root", 

  #  "rfio:/castor/cern.ch/user/f/fabozzi/origZmumuSubSkim.root"
    #"rfio:/castor/cern.ch/user/f/fabozzi/350ZmumuSubSkim.root"
   # 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_4_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0007/CAE2081C-48B5-DE11-9161-001D09F29321.root',
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

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesNtuples_cff")

### output module
##process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesNtuplesOutputModule_cff")


## process.eventInfo = cms.OutputModule (
##     "AsciiOutputModule"
## )

 
## process.goodZToMuMuPath = cms.Path(
##     process.goodZToMuMuSequence 
## )

## process.goodZToMuMu2HLTPath=cms.Path(
##  process.goodZToMuMu2HLTSequence 
##  )

## process.goodZToMuMu1HLTPath=cms.Path(
##  process.goodZToMuMu1HLTSequence 
##  )

## process.goodZToMuMuSameChargePath=cms.Path(
##     process.goodZToMuMuSameChargeSequence
##     )

## process.goodZToMuMuSameCharge2HLTPath=cms.Path(
##     process.goodZToMuMuSameCharge2HLTSequence
##     )

## process.goodZToMuMuSameCharge1HLTPath=cms.Path(
##     process.goodZToMuMuSameCharge1HLTSequence
##     )

    
## process.nonIsolatedZToMuMuPath = cms.Path (
##     process.nonIsolatedZToMuMuSequence
## )

## process.oneNonIsolatedZToMuMuPath = cms.Path(
##     process.oneNonIsolatedZToMuMuSequence 
## )

## process.twoNonIsolatedZToMuMuPath = cms.Path(
##     process.twoNonIsolatedZToMuMuSequence
##     )

## process.goodZToMuMuOneStandAloneMuonPath = cms.Path(
##     process.goodZToMuMuOneStandAloneMuonSequence
##     )

## process.goodZToMuMuOneTrackPath = cms.Path(
##     process.goodZToMuMuOneTrackSequence
##     )


## process.endPath = cms.EndPath( 
##    process.eventInfo +
##    process.NtuplesOut 
##  + process.VtxedNtuplesOut 

## )


