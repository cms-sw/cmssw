import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hiTrackingDqmClientHI = DQMEDHarvester("TrackingDQMClientHeavyIons",
                                              FolderName = cms.string('Tracking/TrackParameters/GeneralProperties')
                                              )

from DQM.TrackingMonitor.TrackFoldedOccupancyClient_cfi import TrackerMapFoldedClient_heavyionTk
hiTrackingDqmClientHeavyIons=cms.Sequence(hiTrackingDqmClientHI*TrackerMapFoldedClient_heavyionTk)
