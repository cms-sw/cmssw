import FWCore.ParameterSet.Config as cms

hltGetRaw = cms.EDFilter("HLTGetRaw",
    RawDataCollection = cms.InputTag("rawDataCollector")
)


