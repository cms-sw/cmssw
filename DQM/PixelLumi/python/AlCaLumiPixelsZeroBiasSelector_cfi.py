import FWCore.ParameterSet.Config as cms

# Selects only AlCaLumiPixels zero-bias events.

alca_lumi_pixels_zerobias_selector = cms.EDFilter("HLTHighLevel",
                                                  TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                                                  HLTPaths = cms.vstring("AlCa_LumiPixels_v*"),
                                                  eventSetupPathsKey = cms.string(''),
                                                  andOr = cms.bool(True),
                                                  throw = cms.bool(False)
                                                  )
