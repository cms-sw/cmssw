import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *

from DQMOffline.JetMET.metDQMConfig_cff     import *
from DQMOffline.JetMET.jetAnalyzer_cff   import *
from DQMOffline.JetMET.caloTowers_cff       import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi import *
from DQMOffline.JetMET.SUSYDQMAnalyzer_cfi  import *
from DQMOffline.JetMET.goodOfflinePrimaryVerticesDQM_cfi import *
AnalyzeBeamHalo.StandardDQM = cms.bool(True)

towerSchemeBAnalyzer.AllHist = cms.untracked.bool(False)

jetMETDQMOfflineSource = cms.Sequence(HBHENoiseFilterResultProducer*goodOfflinePrimaryVerticesDQM*analyzecaloTowersDQM*AnalyzeSUSYDQM*jetDQMAnalyzerSequence*METDQMAnalyzerSequence)
