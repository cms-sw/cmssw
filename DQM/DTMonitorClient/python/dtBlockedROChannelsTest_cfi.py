import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

blockedROChannelTest = DQMEDHarvester("DTBlockedROChannelsTest",
                                      offlineMode = cms.untracked.bool(False),
                                      diagnosticPrescale = cms.untracked.int32(1)
                                      )


