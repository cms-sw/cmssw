import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmEnvSiPixelQuality = DQMEDAnalyzer('DQMEventInfo',
                                     subSystemFolder = cms.untracked.string('PixelPhase1')
                                    )
                            
