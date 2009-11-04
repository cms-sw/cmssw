import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = 
cms.string('MBUEAnalysisRootFile.root')
)

process.genParticles.abortOnUnknownPDGCode = False

process.MessageLogger = cms.Service("MessageLogger",
   
   cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('ERROR')
#ls
#threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
      #                      fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_2_4/RelValMinBias_2M_PROD/GEN-SIM-RECO/MC_31X_V5-v1/0010/34A6537C-0F84-DE11-BD90-001617E30F50.root',
   #     '/store/relval/CMSSW_3_2_4/RelValMinBias_2M_PROD/GEN-SIM-RECO/MC_31X_V5-v1/0009/D0E36D2D-C583-DE11-A8FF-001D09F2426D.root',
 #       '/store/relval/CMSSW_3_2_4/RelValMinBias_2M_PROD/GEN-SIM-RECO/MC_31X_V5-v1/0009/34DA0EF9-C683-DE11-BC85-001D09F231C9.root')
    fileNames =cms.untracked.vstring('file:/afs/cern.ch/user/l/lucaroni/scratch0/QCDSkim/CMSSW_3_2_4/src/QCDAnalysis/UEAnalysis/test/crab_0_090828_171426/res/lowpTRECOSkim_1.root')
,
#skipEvents = cms.untracked.uint32(2)
)

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks
#process.p1 = cms.Path(process.UEAnalysisTracks
          #           + process.UEAnalysisJetsOnlyReco
                      +process.UEAnalysisJets
                      +process.UEAnalysis)


process.UEAnalysisRootple.OnlyRECO     = False
#process.UEAnalysisRootple.OnlyRECO     = True
#process.UEAnalysisRootple500.OnlyRECO  = True
process.UEAnalysisRootple500.OnlyRECO  = False
#process.UEAnalysisRootple1500.OnlyRECO = True
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





