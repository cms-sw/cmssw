import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnalysisRootFileOnlyReco.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/4/6/RelVal-RelValMinBias-1207410101-HLT/0000/18C494D2-0B04-DD11-BF7C-000423D6B5C4.root')
)

process.p1 = cms.Path(process.UEAnalysisTracks*process.UEAnalysisJetsOnlyReco*process.UEAnalysis)

