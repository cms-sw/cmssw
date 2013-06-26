
import FWCore.ParameterSet.Config as cms


DQMMessageLogger = cms.EDAnalyzer("DQMMessageLogger",
                             Categories = cms.vstring(),
                             Directory = cms.string("MessageLogger")
                             )


