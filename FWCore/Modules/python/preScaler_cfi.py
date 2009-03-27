import FWCore.ParameterSet.Config as cms

preScaler = cms.EDFilter("Prescaler",
                         prescaleFactor = cms.int32(1),
                         prescaleOffset = cms.int32(0)
                         )
