import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

SiPixelPhase1ResidualsExtra = DQMEDHarvester("SiPixelPhase1ResidualsExtra",
    TopFolderName = cms.string('PixelPhase1/Tracks/ResidualsExtra'),
    InputFolderName =  cms.string('PixelPhase1/Tracks'),
    MinHits = cms.int32(30)
)
