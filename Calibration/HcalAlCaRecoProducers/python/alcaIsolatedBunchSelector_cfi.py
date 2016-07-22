import FWCore.ParameterSet.Config as cms

# producer for alcaisolatedbunch (HCAL isolated bunch with Jet trigger)
AlcaIsolatedBunchSelector = cms.EDFilter("AlCaIsolatedBunchSelector",
                                         TriggerEventLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                         TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                         ProcessName       = cms.string("HLT"),

                                         TriggerName       = cms.string("HLT_HcalIsolatedBunch"),
                                         )
