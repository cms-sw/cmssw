import FWCore.ParameterSet.Config as cms

hltOfflineReproducibility = cms.EDAnalyzer('HLTOfflineReproducibility',
                                           processNameON = cms.string("HLT"),
                                           processNameOFF = cms.string("TEST"),
                                           triggerTagON  = cms.untracked.InputTag('TriggerResults::HLT'),
                                           triggerTagOFF  = cms.untracked.InputTag('TriggerResults::TEST'),
                                           isRealData = cms.untracked.bool(True),
                                           Nfiles = cms.untracked.int32(235), 
                                           Norm = cms.untracked.double(1.),
                                           LumiSecNumber = cms.untracked.int32(1)                           
                                           )
                                           



