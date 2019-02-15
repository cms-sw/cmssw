import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1Taus.L1TkEGTausAnalyzer_cfi import l1TrkEGTausAnalysis
TkEGEff                 = l1TrkEGTausAnalysis.clone()
TkEGEff.ObjectType      = cms.string("TkEG")
TkEGEff.AnalysisOption  = cms.string("Efficiency")
 
TkEGRate                = l1TrkEGTausAnalysis.clone()
TkEGRate.ObjectType     = cms.string("TkEG")
TkEGRate.AnalysisOption = cms.string("Rate")

