import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet (wantSummary = cms.untracked.bool(False))

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = '101X_dataRun2_Prompt_v11'
process.GlobalTag.globaltag = '101X_dataRun2_HLT_frozen_v10'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery = 500

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    
     'file:$CMSSW_BASE/src/ComparisonPlots/HCALGPUAnalyzer/python/test_bothCPUGPU_NEW.root'
    )
)

process.comparisonPlots = cms.EDAnalyzer('HCALGPUAnalyzer')

process.TFileService = cms.Service('TFileService', fileName = cms.string('test_both.root') )

process.p = cms.Path(process.comparisonPlots)
