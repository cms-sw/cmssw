import FWCore.ParameterSet.Config as cms

from  DQM.SiPixelPhase1Common.SiPixelPhase1OfflineDQM_source_cff import *

siPixelPhase1OfflineDQM_harvesting = cms.Sequence(SiPixelPhase1DigisHarvester + SiPixelPhase1ClustersHarvester)
