import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hiTrackingDqmClientHeavyIons = DQMEDHarvester("TrackingDQMClientHeavyIons",
                                              FolderName = cms.string('Tracking/TrackParameters/GeneralProperties')
                                              )
