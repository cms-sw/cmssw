import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")
process.load("PhysicsTools.HepMCCandAlgos.genEventKTValue_cfi")
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
#        threshold = cms.untracked.string('ERROR')
        threshold = cms.untracked.string('DEBUG')
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
                      +process.UEAnalysisJets
                      +process.genEventKTValue
                      +process.UEAnalysis)

process.UEAnalysisRootple.OnlyRECO     = False
process.UEAnalysisRootple500.OnlyRECO  = False
process.UEAnalysisRootple1500.OnlyRECO = False

process.UEAnalysisRootple.GenJetCollectionName      = 'ueSisCone5GenJet'
process.UEAnalysisRootple.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet'
process.UEAnalysisRootple.TracksJetCollectionName   = 'ueSisCone5TracksJet'
process.UEAnalysisRootple.RecoCaloJetCollectionName = 'sisCone5CaloJets'
process.UEAnalysisRootple500.GenJetCollectionName      = 'ueSisCone5GenJet500'
process.UEAnalysisRootple500.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet500'
process.UEAnalysisRootple500.TracksJetCollectionName   = 'ueSisCone5TracksJet500'
process.UEAnalysisRootple500.RecoCaloJetCollectionName = 'sisCone5CaloJets'
process.UEAnalysisRootple1500.GenJetCollectionName      = 'ueSisCone5GenJet1500'
process.UEAnalysisRootple1500.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet1500'
process.UEAnalysisRootple1500.TracksJetCollectionName   = 'ueSisCone5TracksJet1500'
process.UEAnalysisRootple1500.RecoCaloJetCollectionName = 'sisCone5CaloJets'

#/// Pythia: genEventScale = cms.InputTag("genEventScale")
#/// Herwig: genEventScale = cms.InputTag("genEventKTValue")
process.UEAnalysisRootple.genEventScale     = 'genEventKTValue'
process.UEAnalysisRootple500.genEventScale  = 'genEventKTValue'
process.UEAnalysisRootple1500.genEventScale = 'genEventKTValue'


