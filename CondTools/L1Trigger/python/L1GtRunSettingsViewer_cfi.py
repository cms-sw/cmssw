import FWCore.ParameterSet.Config as cms

L1GtRunSettingsViewer = cms.EDAnalyzer("L1GtRunSettingsViewer",
                                       prescalesKey = cms.string(""),
                                       maskAlgoKey = cms.string(""),
                                       maskTechKey = cms.string(""),
                                       maskVetoAlgoKey = cms.string(""),
                                       maskVetoTechKey = cms.string("")
                                       )
