import FWCore.ParameterSet.Config as cms

# Selects only random-trigger events.

random_trigger_selector = cms.EDFilter("HLTHighLevel",
                                       TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                                       HLTPaths = cms.vstring("AlCa_LumiPixels_Random*"),
                                       eventSetupPathsKey = cms.string(''),
                                       andOr = cms.bool(True),
                                       throw = cms.bool(False)
                                       )
