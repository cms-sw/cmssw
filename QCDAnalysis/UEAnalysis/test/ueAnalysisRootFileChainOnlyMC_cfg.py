import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootpleOnlyMC_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnalysisRootFileOnlyMC.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/CSA08/JetET20/RECO/STARTUP_V2_v1/0017/FE817C29-8A1B-DD11-835D-000423D992DC.root')
)

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisJetsOnlyMC*process.UEAnalysisOnlyMC)

