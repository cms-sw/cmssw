import FWCore.ParameterSet.Config as cms

hltGetRaw = cms.EDAnalyzer("HLTGetRaw",
    RawDataCollection = cms.InputTag("rawDataCollector")
)
