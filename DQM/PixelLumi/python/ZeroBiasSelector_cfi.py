import FWCore.ParameterSet.Config as cms

# Selects only zerobias events.

zerobias_selector = cms.EDFilter("HLTHighLevel",
                                 TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                                 HLTPaths = cms.vstring("AlCa_LumiPixels_ZeroBias*"),
                                 eventSetupPathsKey = cms.string(''),
                                 andOr = cms.bool(True),
                                 throw = cms.bool(False)
                                 )
