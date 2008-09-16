import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnalysisRootFileOnlyReco.root')
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
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/CSA08/MinBias/ALCARECO/STARTUP_V2_TkAlMinBias_v1/0016/002E3EE3-A71A-DD11-95A0-001617E30F4C.root')
)

process.p1 = cms.Path(process.UEAnalysisTracks*process.UEAnalysisJetsOnlyReco*process.UEAnalysis)
process.selectTracks.src = 'ALCARECOTkAlMinBias'

