import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzer_cff import *
from DQMOffline.JetMET.caloTowers_cff     import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi import *

AnalyzeBeamHalo.StandardDQM = cms.bool(True)

jetMETAnalyzer.DoPFJetAnalysis  = cms.untracked.bool(False)
jetMETAnalyzer.DoJPTJetAnalysis = cms.untracked.bool(False)
jetMETAnalyzer.DoIterativeCone  = cms.untracked.bool(False)
jetMETAnalyzer.DoDiJetSelection = cms.untracked.bool(False)

jetMETAnalyzer.AKJetsCollectionLabel  = cms.InputTag("iterativeConePu5CaloJets")
jetMETAnalyzer.ICJetsCollectionLabel  = cms.InputTag("iterativeConePu5CaloJets")

jetMETAnalyzer.DoPFJetCleaning  = cms.untracked.bool(False)
jetMETAnalyzer.DoJPTJetCleaning = cms.untracked.bool(False)
jetMETAnalyzer.DoJetCleaning    = cms.untracked.bool(False)

jetMETAnalyzer.DoCaloMETAnalysis            = cms.untracked.bool(False)
jetMETAnalyzer.DoTcMETAnalysis              = cms.untracked.bool(False)
jetMETAnalyzer.DoMuCorrMETAnalysis          = cms.untracked.bool(False)
jetMETAnalyzer.DoPfMETAnalysis              = cms.untracked.bool(False)
jetMETAnalyzer.DoHTMHTAnalysis              = cms.untracked.bool(False)

jetMETAnalyzer.caloMETAnalysis.allHist                = cms.bool(False)

jetMETAnalyzer.caloMETAnalysis.cleanupSelection       = cms.bool(False)
jetMETAnalyzer.caloMETNoHFAnalysis.cleanupSelection   = cms.bool(False)
jetMETAnalyzer.caloMETHOAnalysis.cleanupSelection     = cms.bool(False)
jetMETAnalyzer.caloMETNoHFHOAnalysis.cleanupSelection = cms.bool(False)
jetMETAnalyzer.pfMETAnalysis.cleanupSelection         = cms.bool(False)
jetMETAnalyzer.tcMETAnalysis.cleanupSelection         = cms.bool(False)

# prevent jetIDHelper from calculating cell based variables which don't make sense w/ HI bkg subtraction
jetMETAnalyzer.jetAnalysis.JetIDParams.useRecHits = False
jetMETAnalyzer.CleanedjetAnalysis.JetIDParams.useRecHits = False

towerSchemeBAnalyzer.AllHist = cms.untracked.bool(False)

jetMETDQMOfflineSource = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerSequence)
#jetMETDQMOfflineSource = cms.Sequence(analyzecaloTowersDQM*AnalyzeBeamHalo*jetMETAnalyzerSequence)
