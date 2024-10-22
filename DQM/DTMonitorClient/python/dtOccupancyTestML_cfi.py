import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtOccupancyTestML = DQMEDHarvester("DTOccupancyTestML",
                                   testPulseMode = cms.untracked.bool(False),
                                   runOnAllHitsOccupancies = cms.untracked.bool(True),
                                   runOnNoiseOccupancies = cms.untracked.bool(False),
                                   runOnInTimeOccupancies = cms.untracked.bool(False),
                                   nEventsCert = cms.untracked.int32(2500),
                                   nEventsZeroPC = cms.untracked.int32(10),
                                   nEventsMinPC = cms.untracked.int32(2200)
                                   )


