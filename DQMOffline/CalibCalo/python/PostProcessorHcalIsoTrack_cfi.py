import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

PostProcessorHcalIsoTrack = cms.DQMEDHarvester("DQMHcalIsoTrackPostProcessor",
     subDir = cms.untracked.string("AlCaReco/HcalIsoTrack"),
)
