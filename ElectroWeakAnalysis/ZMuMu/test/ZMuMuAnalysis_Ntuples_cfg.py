import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuNtp")



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
    input = cms.untracked.int32(10)
)



process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     "rfio:/castor/cern.ch/user/d/degrutto/zmmWithBsPv/testZMuMuSubSkim_1.root", 
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
    fileName = cms.string("ewkZMuMuCategoriesTest.root")
)


### vertexing
#process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesVtxed_cff")

### plots

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesPlots_cff")

### ntuple

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisNtupler_cff")

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


#process.endPath = cms.EndPath( 
##    process.eventInfo +
#    process.NtuplesOut 
    ##  + process.VtxedNtuplesOut 
#    )


