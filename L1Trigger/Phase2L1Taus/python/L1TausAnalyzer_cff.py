import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1Taus.L1TausAnalyzer_cfi import l1TausAnalysis
TrkTauEff                 = l1TausAnalysis.clone()
TrkTauEff.ObjectType      = cms.string("TrkTau")
TrkTauEff.GenEtaCutOff    = cms.double(2.3)
TrkTauEff.EtaCutOff       = cms.double(2.5)
TrkTauEff.AnalysisOption  = cms.string("Efficiency")
 
TrkTauRate                = l1TausAnalysis.clone()
TrkTauRate.ObjectType     = cms.string("TrkTau")
TrkTauRate.GenEtaCutOff   = cms.double(2.3)
TrkTauRate.EtaCutOff      = cms.double(2.5)
TrkTauRate.AnalysisOption = cms.string("Rate")

TkEGEff                 = l1TausAnalysis.clone()
TkEGEff.ObjectType      = cms.string("TkEG")
TkEGEff.AnalysisOption  = cms.string("Efficiency")
 
TkEGRate                = l1TausAnalysis.clone()
TkEGRate.ObjectType     = cms.string("TkEG")
TkEGRate.AnalysisOption = cms.string("Rate")


