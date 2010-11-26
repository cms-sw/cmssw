import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzerCosmic_cff import *
from DQMOffline.JetMET.caloTowers_cff           import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi     import *

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

towerSchemeBAnalyzer.AllHist = cms.untracked.bool(False)

jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*AnalyzeBeamHalo*jetMETAnalyzerCosmicSequence)
#jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerCosmicSequence)
