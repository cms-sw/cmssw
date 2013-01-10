import FWCore.ParameterSet.Config as cms

hltOfflineReproducibilityDQM = cms.EDAnalyzer('HLTOfflineReproducibilityDQM',
                                           processNameORIG = cms.string("HLT"),
                                           processNameNEW = cms.string("TEST"),
                                           triggerTagORIG  = cms.untracked.InputTag('TriggerResults::HLT'),
                                           triggerTagNEW  = cms.untracked.InputTag('TriggerResults::TEST'),
                                           isRealData = cms.untracked.bool(True),
                                           Nfiles = cms.untracked.int32(235), 
                                           Norm = cms.untracked.double(1.),
                                           LumiSecNumber = cms.untracked.int32(1)                           
                                           )
                                           



