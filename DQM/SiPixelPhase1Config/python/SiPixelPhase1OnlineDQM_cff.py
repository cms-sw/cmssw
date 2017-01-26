import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

StandardSpecifications1D.append(
    Specification(PerLayer1D).groupBy("PXBarrel/PXLayer/OnlineBlock") # per-layer with history for online
                             .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
                             .save()
)

StandardSpecifications1D.append(
    Specification(PerLayer1D).groupBy("PXForward/PXDisk/OnlineBlock") # per-layer with history for online
                             .groupBy("PXForward/PXDisk", "EXTEND_Y")
                             .save()
)

StandardSpecifications1D.append(
    Specification().groupBy("PXBarrel/OnlineBlock") # per-layer with history for online
                   .groupBy("PXBarrel", "EXTEND_Y")
                   .save()
)
StandardSpecifications1D.append(
    Specification().groupBy("PXForward/OnlineBlock") # per-layer with history for online
                   .groupBy("PXForward", "EXTEND_Y")
                   .save()
)

StandardSpecifications1D_Num.append(
    Specification(PerLayer1D).groupBy("PXBarrel/PXLayer/OnlineBlock/DetId/Event") # per-layer with history for online
                             .reduce("COUNT")
                             .groupBy("PXBarrel/PXLayer/OnlineBlock") 
                             .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
                             .save()
)
StandardSpecifications1D_Num.append(
    Specification(PerLayer1D).groupBy("PXForward/PXDisk/OnlineBlock/DetId/Event") # per-layer with history for online
                             .reduce("COUNT")
                             .groupBy("PXForward/PXDisk/OnlineBlock") 
                             .groupBy("PXForward/PXDisk", "EXTEND_Y")
                             .save()
)
StandardSpecifications1D_Num.append(
    Specification().groupBy("PXBarrel/OnlineBlock/DetId/Event") # per-layer with history for online
                   .reduce("COUNT")
                   .groupBy("PXBarrel/OnlineBlock") 
                   .groupBy("PXBarrel", "EXTEND_Y")
                   .save()
)
StandardSpecifications1D_Num.append(
    Specification().groupBy("PXForward/OnlineBlock/DetId/Event") # per-layer with history for online
                   .reduce("COUNT")
                   .groupBy("PXForward/OnlineBlock") 
                   .groupBy("PXForward", "EXTEND_Y")
                   .save()
)

# Configure Phase1 DQM for Phase0 data
SiPixelPhase1Geometry.n_inner_ring_blades = 24 # no outer ring

# Turn on 'online' harvesting. This has to be set before other configs are 
# loaded (due to how the DefaultHisto PSet is later cloned), therefore it is
# here and not in the harvestng config.
DefaultHisto.perLumiHarvesting = True
DefaultHistoDigiCluster.perLumiHarvesting = True
DefaultHistoSummary.perLumiHarvesting = True
DefaultHistoTrack.perLumiHarvesting = True


# Pixel Digi Monitoring
from DQM.SiPixelPhase1Digis.SiPixelPhase1Digis_cfi import *
SiPixelPhase1DigisAnalyzer.src = cms.InputTag("siPixelDigis") # adapt for real data

# Cluster (track-independent) monitoring
from DQM.SiPixelPhase1Clusters.SiPixelPhase1Clusters_cfi import *

# We could overwrite the Harvesters like this, and use the custom() steps to
# perform resetting of histograms.
#SiPixelPhase1ClustersHarvester = cms.EDAnalyzer("SiPixelPhase1OnlineHarvester",
#    histograms = SiPixelPhase1ClustersConf,
#    geometry = SiPixelPhase1Geometry
#)


# Raw data errors
from DQM.SiPixelPhase1RawData.SiPixelPhase1RawData_cfi import *

from DQM.SiPixelPhase1Common.SiPixelPhase1GeometryDebug_cfi import *

PerModule.enabled = True

siPixelPhase1OnlineDQM_source = cms.Sequence(
   SiPixelPhase1DigisAnalyzer
 + SiPixelPhase1ClustersAnalyzer
 + SiPixelPhase1RawDataAnalyzer
# + SiPixelPhase1GeometryDebugAnalyzer
)

siPixelPhase1OnlineDQM_harvesting = cms.Sequence(
   SiPixelPhase1DigisHarvester 
 + SiPixelPhase1ClustersHarvester
 + SiPixelPhase1RawDataHarvester
# + SiPixelPhase1GeometryDebugHarvester
)
