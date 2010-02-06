import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzerCosmic_cff import *
from DQMOffline.JetMET.caloTowers_cff           import *

jetMETDQMOfflineSourceCosmic = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerCosmicSequence)
