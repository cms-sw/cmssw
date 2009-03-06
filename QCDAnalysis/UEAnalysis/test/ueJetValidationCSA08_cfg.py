import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")
process.load("JetMETCorrections.Configuration.L2L3Corrections_iCSA08_S156_cff")
process.prefer("L2L3JetCorrectorScone5")
process.load("QCDAnalysis.UEAnalysis.UEJetValidationCSA08_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('UEJetValidation.root')
)

process.MessageLogger = cms.Service("MessageLogger",
                                    cerr = cms.untracked.PSet(
    default = cms.untracked.PSet(
    limit = cms.untracked.int32(10)
    )
    ),
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
                            fileNames = cms.untracked.vstring('file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/RelValQCD_Pt_80_120-Ideal-000AD2A4-6E86-DD11-AA99-000423D9863C.root')
                            )

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets*process.L2L3CorJetScone5+process.UEJetValidation)




