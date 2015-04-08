import FWCore.ParameterSet.Config as cms

hltobject = cms.EDAnalyzer("TriggerObjectAnalyzer",
                           processName = cms.string("HLT"),
                           triggerNames = cms.vstring("HLT_HIJet80_v1", "HLT_HIJet65_v1", "HLT_HIJet55_v1"),
                           triggerResults = cms.InputTag("TriggerResults","","HLT"),
                           triggerEvent   = cms.InputTag("hltTriggerSummaryAOD","","HLT")
                           )
