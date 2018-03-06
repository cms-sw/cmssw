import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

GEMDQMHarvester = DQMEDHarvester("GEMDQMHarvester", 
  Name = cms.untracked.string('HarvestingAnalyzer'),

)
