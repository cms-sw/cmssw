import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")
#process.load("QCDAnalysis.UEAnalysis.UERegionSelector_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnalysisRootFile.root')
)

process.MessageLogger = cms.Service("MessageLogger",
                               
                                    cout = cms.untracked.PSet(
    #threshold = cms.untracked.string('ERROR')
     threshold = cms.untracked.string('DEBUG')
    ),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre11/RelValMinBias_2M_PROD/GEN-SIM-RECO/MC_31X_V1-v1/0000/E49ABFE2-EE64-DE11-A43C-000423D991D4.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias_2M_PROD/GEN-SIM-RECO/MC_31X_V1-v1/0000/D497326B-C564-DE11-AE6A-000423D95030.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValMinBias_2M_PROD/GEN-SIM-RECO/MC_31X_V1-v1/0000/A65659E9-C664-DE11-847D-001D09F295A1.root'
)
                            )

#process.EventAnalyzer = cms.EDAnalyzer("EventContentAnalyzer")
#process.UEAnalysisEventContent = cms.OutputModule("PoolOutputModule",
#                                                  fileName = cms.untracked.string('UEAnalysisEventContent.root'),
#                                                  outputCommands = cms.untracked.vstring('keep *')
#                                                  )

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UEAnalysis)
#process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UEAnalysis+process.UERegionSelector)
process.UEAnalysisRootple.OnlyRECO = False
process.UEAnalysisRootple500.OnlyRECO = False
process.UEAnalysisRootple1500.OnlyRECO = False


