import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzerCosmic_cff import *
from DQMOffline.JetMET.caloTowers_cff           import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi     import *

jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*AnalyzeBeamHalo*jetMETAnalyzerSequence)
#jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerCosmicSequence)
