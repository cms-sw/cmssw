import FWCore.ParameterSet.Config as cms

IsoTracksProdFilter = cms.EDFilter("AlCaIsoTracksProducerFilter",
                                   TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                   Triggers          = cms.vstring("HLT_IsoTrackHB","HLT_IsoTrackHE"),
                                   ProcessName       = cms.string("HLT"),
)
