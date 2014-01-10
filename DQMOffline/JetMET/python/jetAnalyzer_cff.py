import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *



jetAnalyzerSequence = cms.Sequence(jetAnalyzerAk5CaloUncleaned*jetAnalyzerAk5CaloCleaned*jetAnalyzerAk5CaloDiJetCleaned
                                   *jetAnalyzerAk5JPTCleaned
                                   *jetAnalyzerAk5PFUncleaned*jetAnalyzerAk5PFCleaned*jetAnalyzerAk5PFDiJetCleaned)

