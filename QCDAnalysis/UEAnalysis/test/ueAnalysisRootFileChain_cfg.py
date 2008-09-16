import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnalysisRootFile.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        )
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/CSA08/JetET80/GEN-SIM-RECO/CSA08_S156_v1/0066/1A0E344B-4D2C-DD11-91E7-001731AF686B.root')
)

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks*process.UEAnalysisJets*process.UEAnalysis)
process.UEAnalysisRootple.OnlyRECO = False
process.UEAnalysisRootple500.OnlyRECO = False

