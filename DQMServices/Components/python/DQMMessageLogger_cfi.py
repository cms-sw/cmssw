
import FWCore.ParameterSet.Config as cms


DQMMessageLogger = DQMStep1Module('DQMMessageLogger',
                             Categories = cms.vstring(),
                             Directory = cms.string("MessageLogger")
                             )


