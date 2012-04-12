import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzerCosmic_cff import *
from DQMOffline.JetMET.caloTowers_cff           import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi     import *
from DQMOffline.JetMET.SUSYDQMAnalyzer_cfi import *

AnalyzeBeamHalo.StandardDQM = cms.bool(True)

towerSchemeBAnalyzer.AllHist = cms.untracked.bool(False)

jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*AnalyzeSUSYDQM*jetMETAnalyzerCosmicSequence)
#jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerCosmicSequence)
#jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*AnalyzeBeamHalo*AnalyzeSUSYDQM*jetMETAnalyzerCosmicSequence)
