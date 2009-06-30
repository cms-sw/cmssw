import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETAnalyzer_cff import *
from DQMOffline.JetMET.caloTowers_cff     import *

jetMETDQMOfflineSource = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerSequence)
