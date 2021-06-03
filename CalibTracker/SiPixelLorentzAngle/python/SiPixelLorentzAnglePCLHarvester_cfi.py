import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

SiPixelLorentzAnglePCLHarvester = DQMEDHarvester(
    "SiPixelLorentzAnglePCLHarvester",
    dqmDir = cms.string('AlCaReco/SiPixelLorentzAngle')
)
