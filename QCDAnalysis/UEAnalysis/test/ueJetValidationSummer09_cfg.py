import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")


process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer09_cff")
process.prefer("L2L3JetCorrectorSC5Calo")

process.load("QCDAnalysis.UEAnalysis.UEJetValidation_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('UEJetValidation.root')
)

process.MessageLogger = cms.Service("MessageLogger",
                                  
                                    cout = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR')
    #threshold = cms.untracked.string('DEBUG')
    ),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_2/RelValMinBias_2M_PROD/GEN-SIM-RECO/MC_31X_V3-v1/0007/E03C0F5F-B278-DE11-AAC0-001D09F291D2.root')
                            ,skipEvents = cms.untracked.uint32(0) 
                             )

#process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets*process.UEJetValidation)
process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets*process.L2L3CorJetSC5Calo+process.UEJetValidation)





