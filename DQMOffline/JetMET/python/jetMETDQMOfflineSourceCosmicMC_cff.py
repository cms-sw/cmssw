import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzerCosmic_cff import *
from DQMOffline.JetMET.caloTowers_cff           import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi     import *

AnalyzeBeamHalo.StandardDQM = cms.bool(True)

jetMETAnalyzer.DoPFJetCleaning  = cms.untracked.bool(False)
jetMETAnalyzer.DoJPTJetCleaning = cms.untracked.bool(False)
jetMETAnalyzer.DoJetCleaning    = cms.untracked.bool(False)

jetMETAnalyzer.caloMETAnalysis.allHist                = cms.bool(False)

jetMETAnalyzer.caloMETAnalysis.cleanupSelection       = cms.bool(False)
jetMETAnalyzer.caloMETNoHFAnalysis.cleanupSelection   = cms.bool(False)
jetMETAnalyzer.caloMETHOAnalysis.cleanupSelection     = cms.bool(False)
jetMETAnalyzer.caloMETNoHFHOAnalysis.cleanupSelection = cms.bool(False)
jetMETAnalyzer.pfMETAnalysis.cleanupSelection         = cms.bool(False)
jetMETAnalyzer.tcMETAnalysis.cleanupSelection         = cms.bool(False)

towerSchemeBAnalyzer.AllHist = cms.untracked.bool(False)

jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerCosmicSequence)
#jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*AnalyzeBeamHalo*jetMETAnalyzerCosmicSequence)
