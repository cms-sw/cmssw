import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
dqmEnvSiPixelQuality = DQMEDHarvester('DQMHarvestingMetadata',
                                      subSystemFolder = cms.untracked.string('PixelPhase1')
                                      )
                            
