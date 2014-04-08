import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *

from DQMOffline.JetMET.metDQMConfig_cff     import *
from DQMOffline.JetMET.jetAnalyzer_cff   import *
from DQMOffline.JetMET.caloTowers_cff       import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi import *
from DQMOffline.JetMET.SUSYDQMAnalyzer_cfi  import *

AnalyzeBeamHalo.StandardDQM = cms.bool(True)

towerSchemeBAnalyzer.AllHist = cms.untracked.bool(False)

jetDQMAnalyzerAk5CaloUncleaned.runcosmics = cms.untracked.bool(True)
 
caloMetDQMAnalyzer.runcosmics = cms.untracked.bool(True)

jetMETDQMOfflineSourceCosmic = cms.Sequence(HBHENoiseFilterResultProducer*analyzecaloTowersDQM*AnalyzeSUSYDQM*METAnalyzerSequenceCosmics*jetDQMAnalyzerSequenceCosmics*METDQMAnalyzerSequenceCosmics)

#jetMETDQMOfflineSource = cms.Sequence(analyzecaloTowersDQM*jetMETAnalyzerSequence)
#jetMETDQMOfflineSource = cms.Sequence(analyzecaloTowersDQM*AnalyzeBeamHalo*jetMETAnalyzerSequence)
