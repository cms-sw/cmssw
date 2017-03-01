import FWCore.ParameterSet.Config as cms

process = cms.Process('GETGBR')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Run2015D/DoubleEG/MINIAOD/PromptReco-v4/000/258/159/00000/027612B0-306C-E511-BD47-02163E014496.root'),
)

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '74X_dataRun2_Prompt_v4', '')

process.getGBR25ns = cms.EDAnalyzer("GBRForestGetterFromDB",
    grbForestName = cms.string("gedelectron_p4combination_25ns"),
    outputFileName = cms.untracked.string("GBRForest_data_25ns.root"),
)
    
process.getGBR50ns = cms.EDAnalyzer("GBRForestGetterFromDB",
    grbForestName = cms.string("gedelectron_p4combination_50ns"),
    outputFileName = cms.untracked.string("GBRForest_data_50ns.root"),
)

process.path = cms.Path(
    process.getGBR25ns +
    process.getGBR50ns
)
