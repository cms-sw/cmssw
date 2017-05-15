import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

PostProcessorHcalIsoTrack = cms.DQMDQMEDProducer("DQMHcalIsoTrackPostProcessor",
     subDir = cms.untracked.string("AlCaReco/HcalIsoTrack"),
)
