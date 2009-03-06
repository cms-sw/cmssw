import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEJetMultiplicity")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsKt_cfi")
process.load("PhysicsTools.HepMCCandAlgos.genEventKTValue_cfi")
process.load("QCDAnalysis.UEAnalysis.UEJetMultiplicitySummer08_cfi")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('UEJetMultiplicity.root')
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
#                            fileNames = cms.untracked.vstring('/store/mc/Summer08/HerwigQCDPt15/GEN-SIM-RECO/IDEAL_V9_v1/0000/04631626-4493-DD11-A07F-00D0680BF8C7.root')                            
                            fileNames = cms.untracked.vstring('/store/mc/Summer08/HerwigQCDPt170/GEN-SIM-RECO/IDEAL_V9_v1/0005/661F442D-30A4-DD11-B881-00145E55647F.root')
                            )

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks
                      +process.UEAnalysisJetsKt
                      +process.genEventKTValue
                      +process.UEJetMultiplicitySummer08)

process.JetMultiplicity_0_pT_15.genEventScale    = 'genEventKTValue'
process.JetMultiplicity_15_pT_30.genEventScale   = 'genEventKTValue'
process.JetMultiplicity_30_pT_80.genEventScale   = 'genEventKTValue'
process.JetMultiplicity_80_pT_170.genEventScale  = 'genEventKTValue'
process.JetMultiplicity_170_pT_300.genEventScale = 'genEventKTValue'
process.JetMultiplicity_300_pT_470.genEventScale = 'genEventKTValue'

