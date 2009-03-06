import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer08_cff")
process.prefer("L2L3JetCorrectorSC5Calo")
process.load("PhysicsTools.HepMCCandAlgos.genEventKTValue_cfi")
process.load("QCDAnalysis.UEAnalysis.UEJetValidationSummer08_cfi")
#process.load("QCDAnalysis.UEAnalysis.UEJetValidationSummer08QCDDijet_cfi")

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
                            fileNames = cms.untracked.vstring('/store/mc/Summer08/HerwigQCDPt15/GEN-SIM-RECO/IDEAL_V9_v1/0000/04631626-4493-DD11-A07F-00D0680BF8C7.root')                            
                            )

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks
                      +process.UEAnalysisJets*process.L2L3CorJetSC5Calo
                      +process.genEventKTValue
                      +process.UEJetValidation)

process.Summer08_0_pT_15_threshold900.genEventScale    = 'genEventKTValue'
process.Summer08_15_pT_30_threshold900.genEventScale   = 'genEventKTValue'
process.Summer08_30_pT_80_threshold900.genEventScale   = 'genEventKTValue'
process.Summer08_80_pT_170_threshold900.genEventScale  = 'genEventKTValue'
process.Summer08_170_pT_300_threshold900.genEventScale = 'genEventKTValue'
process.Summer08_300_pT_470_threshold900.genEventScale = 'genEventKTValue'

