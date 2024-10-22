import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtNoiseAnalysisMonitor = DQMEDHarvester("DTNoiseAnalysisTest",
                                        noisyCellDef = cms.untracked.int32(500), # (in Hz): for cosmics 500 Hz, based on runs 341052 and 339579, variable chamber by chamber for collisions
                                        isCosmics  = cms.untracked.bool(False),
                                        doSynchNoise = cms.untracked.bool(False),
                                        detailedAnalysis = cms.untracked.bool(False),
                                        maxSynchNoiseRate = cms.untracked.double(0.001),
                                        noiseSafetyFactor = cms.untracked.double(5.),
                                        nEventsCert = cms.untracked.int32(1000)
                                        )



