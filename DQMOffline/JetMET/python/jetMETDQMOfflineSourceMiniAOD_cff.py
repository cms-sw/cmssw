import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *

from DQMOffline.JetMET.metDQMConfigMiniAOD_cff     import *
from DQMOffline.JetMET.jetAnalyzerMiniAOD_cff   import *
from DQMOffline.JetMET.BeamHaloAnalyzer_cfi import *
from DQMOffline.JetMET.goodOfflinePrimaryVerticesDQMMiniAOD_cfi import *
AnalyzeBeamHalo.StandardDQM = cms.bool(True)

jetMETDQMOfflineSource = cms.Sequence(HBHENoiseFilterResultProducer*goodOfflinePrimaryVerticesDQM*jetDQMAnalyzerSequence*METDQMAnalyzerSequence)
