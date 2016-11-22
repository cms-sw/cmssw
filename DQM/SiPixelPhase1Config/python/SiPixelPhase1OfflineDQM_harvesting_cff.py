import FWCore.ParameterSet.Config as cms

from  DQM.SiPixelPhase1Config.SiPixelPhase1OfflineDQM_source_cff import *

siPixelPhase1OfflineDQM_harvesting = cms.Sequence(SiPixelPhase1DigisHarvester 
                                                + SiPixelPhase1ClustersHarvester
                                                + SiPixelPhase1RecHitsHarvester
                                                + SiPixelPhase1TrackResidualsHarvester
                                                + SiPixelPhase1TrackClustersHarvester
                                                + SiPixelPhase1TrackEfficiencyHarvester
                                                )

siPixelPhase1OfflineDQM_harvesting_cosmics = siPixelPhase1OfflineDQM_harvesting.copyAndExclude([
   SiPixelPhase1TrackClustersHarvester,
   SiPixelPhase1TrackEfficiencyHarvester,
])
