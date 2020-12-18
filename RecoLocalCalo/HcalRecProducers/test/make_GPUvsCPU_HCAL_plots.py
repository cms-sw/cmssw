import FWCore.ParameterSet.Config as cms

process = cms.Process("PLOT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery = 500

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:GPUvsCPU_HCAL_rechits.root')
)

process.comparisonPlots = cms.EDAnalyzer('HCALGPUAnalyzer')

process.TFileService = cms.Service('TFileService',
    fileName = cms.string('GPUvsCPU_HCAL_plots.root')
)

process.path = cms.Path(process.comparisonPlots)
