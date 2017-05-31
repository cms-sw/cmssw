import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtTPmonitorTest = DQMEDHarvester("DTOccupancyTest",
                                 testPulseMode = cms.untracked.bool(True),
                                 runOnAllHitsOccupancies = cms.untracked.bool(False),
                                 runOnNoiseOccupancies = cms.untracked.bool(False),
                                 runOnInTimeOccupancies = cms.untracked.bool(True)
                                 )

