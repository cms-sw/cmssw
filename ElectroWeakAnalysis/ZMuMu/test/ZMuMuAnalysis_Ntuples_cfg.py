import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuNtupla")



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

    "file:testZMuMuSubskimUserData.root"    

    )
                            )



# replace ZSelection if wanted......
## from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import * 
## zSelection.cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 0")



process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequencesUserData_cff")

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




