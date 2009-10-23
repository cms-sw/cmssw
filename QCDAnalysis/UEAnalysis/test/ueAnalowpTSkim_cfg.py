import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnaSkimRootFile.root')
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.threshold = cms.untracked.string('INFO')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:lowpTRECOSkim.root')#,
#                            intputCommands = cms.untracked.vstring("drop *_goodTracks_*_*")
                            )

#process.allTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
#    src = cms.InputTag("selectTracks"),
#    particleType = cms.string('pi+')
#)

#process.goodTracks = cms.EDFilter("PtMinCandViewSelector",
#    src = cms.InputTag("allTracks"),
#    ptMin = cms.double(0.29)
#)

process.EventAnalyzer = cms.EDAnalyzer("EventContentAnalyzer")


process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UEAnalysis)
#process.p1 = cms.Path(process.goodTracks+process.UEAnalysis)
#process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UEAnalysis+process.UERegionSelector)
#process.UEAnalysisRootple.OnlyRECO = True
#process.UEAnalysisRootple500.OnlyRECO = True
#process.UEAnalysisRootple1500.OnlyRECO = True
process.UEAnalysisRootple.OnlyRECO = False
process.UEAnalysisRootple500.OnlyRECO = False
process.UEAnalysisRootple1500.OnlyRECO = False


#process.UEPoolOutput = cms.EndPath(process.UEAnalysisEventContent)
