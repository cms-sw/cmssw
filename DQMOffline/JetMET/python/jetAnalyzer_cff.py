import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *



jetDQMAnalyzerSequence = cms.Sequence(jetDQMAnalyzerAk5CaloUncleaned*jetDQMAnalyzerAk5CaloCleaned
                                   *jetDQMAnalyzerAk5JPTCleaned
                                   *jetDQMAnalyzerAk5PFUncleaned*jetDQMAnalyzerAk5PFCleaned
                                   )

jetDQMAnalyzerSequenceCosmics = cms.Sequence(jetDQMAnalyzerAk5CaloUncleaned)

jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMAnalyzerIC5CaloHIUncleaned)
