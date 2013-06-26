import FWCore.ParameterSet.Config as cms

DQMMessageLoggerClient = cms.EDAnalyzer ("DQMMessageLoggerClient",
                                   Directory = cms.string("MessageLogger")

    )
