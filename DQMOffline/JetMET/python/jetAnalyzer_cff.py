import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *



jetDQMAnalyzerSequence = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned*jetDQMAnalyzerAk4CaloCleaned
#                                   *jetDQMAnalyzerAk4JPTCleaned
                                   *jetDQMAnalyzerAk4PFUncleaned*jetDQMAnalyzerAk4PFCleaned
                                   )

jetDQMAnalyzerSequenceCosmics = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned)

jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMAnalyzerIC5CaloHIUncleaned)
