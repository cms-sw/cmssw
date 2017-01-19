import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *


#-------------------------------------------------------------------------------
#   Histogram settings
#   Getting directly from SiPixelPhase1Cluster settings
#-------------------------------------------------------------------------------
import DQM.SiPixelPhase1Clusters.SiPixelPhase1Clusters_cfi as clusterset


SiPixelPhase1ClustersFilteredNClusters = clusterset.SiPixelPhase1ClustersNClusters.clone (
  name = "filtered_clusters",
  title = "Filtered Clusters",
)


SiPixelPhase1ClustersFilteredConf = cms.VPSet(
  SiPixelPhase1ClustersFilteredNClusters,
)

SiPixelPhase1ClustersFilteredHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
    histograms = SiPixelPhase1ClustersFilteredConf,
    geometry = SiPixelPhase1Geometry
)

#-------------------------------------------------------------------------------
#   Defining filters to use
#-------------------------------------------------------------------------------
import DQM.SiPixelPhase1Common.TriggerEventFlag_cfi as flagset

SiPixelPhase1ClustersFilteredAnalyzer = cms.EDAnalyzer(
    "SiPixelPhase1ClustersFiltered",
    src = cms.InputTag("siPixelClusters"),
    histograms = SiPixelPhase1ClustersFilteredConf,
    geometry = SiPixelPhase1Geometry,
    flaglist = cms.VPSet(
        flagset.genericTriggerEventFlag4L1bd,
        flagset.genericTriggerEventFlag4HLTdb
    )
)
