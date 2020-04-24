import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import *

from DQMOffline.JetMET.metDQMConfig_cff     import *
from DQMOffline.JetMET.jetAnalyzer_cff   import *

jetMETDQMOfflineSource = cms.Sequence(HBHENoiseFilterResultProducer*jetDQMAnalyzerSequenceHI)
