import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

# this should be moved to RecHits
SiPixelPhase1TrackEfficiencyClusterProb = DefaultHisto.clone(
  name = "clusterprob",
  title = "Cluster Probability",
  xlabel = "log_10(Pr)",
  range_min = -10, range_max = 0, range_nbins = 200,
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping).saveAll()
  )
)

SiPixelPhase1TrackEfficiencyEfficiency = DefaultHisto.clone(
  name = "hitefficiency",
  title = "Hit Efficiency",
  # most stuff done in custom() step, carried over from valid/missing histos
  specs = cms.VPSet(
    Specification().groupBy("").save().custom().save()
  )
)

SiPixelPhase1TrackEfficiencyValid = DefaultHisto.clone(
  bookUndefined = False, # Barrel-only stuff below
  name = "valid",
  title = "Valid Hits",
  xlabel = "Valid Hits",
  dimensions = 0,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultPerModule)
                   .reduce("COUNT")
                   .groupBy(DefaultHisto.defaultGrouping, "EXTEND_X")
                   .save()
                   .custom()
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_Y")
                   .save()
                   .custom(),
    Specification().groupBy("PXBarrel/PXLayer/signedLadder/signedModule")
                   .reduce("COUNT")
                   .groupBy("PXBarrel/PXLayer/signedLadder", "EXTEND_X")
                   .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
                   .save()
                   .custom()
                   .groupBy("PXBarrel", "SUM")
                   .save()
                   .custom(),
    Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/ROCinLadder|ROCinBlade")
                   .reduce("COUNT")
                   .groupBy(DefaultHisto.defaultGrouping, "EXTEND_X")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_Y")
                   .save()
                   .custom()
  )
)

SiPixelPhase1TrackEfficiencyMissing = SiPixelPhase1TrackEfficiencyValid.clone(
  name = "missing",
  title = "Missing Hits",
  xlabel = "Missing Hits",
)

SiPixelPhase1TrackEfficiencyConf = cms.VPSet(
  SiPixelPhase1TrackEfficiencyClusterProb,
  SiPixelPhase1TrackEfficiencyEfficiency,
  SiPixelPhase1TrackEfficiencyValid,
  SiPixelPhase1TrackEfficiencyMissing,
)


SiPixelPhase1TrackEfficiencyAnalyzer = cms.EDAnalyzer("SiPixelPhase1TrackEfficiency",
        clusters = cms.InputTag("siPixelClusters"),
        trajectories = cms.InputTag("generalTracks"),
        primaryvertices = cms.InputTag("offlinePrimaryVertices"),
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackEfficiencyHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)
