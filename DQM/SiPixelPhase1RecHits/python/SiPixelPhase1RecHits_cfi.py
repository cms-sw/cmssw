import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1RecHitsNRecHits = DefaultHisto.clone(
  name = "rechits",
  title = "RecHits",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "rechits",
  dimensions = 0,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/DetId/Event") 
                   .reduce("COUNT") 
                   .groupBy(DefaultHisto.defaultGrouping)
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
    Specification().groupBy(DefaultHisto.defaultGrouping.value() + "/DetId/Event")
                   .reduce("COUNT")
                   .groupBy(parent(DefaultHisto.defaultGrouping))
                   .save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule.value() + "/Event")
                            .reduce("COUNT")
                            .groupBy(DefaultHisto.defaultPerModule)
                            .save()
  )
)

SiPixelPhase1RecHitsClustX = DefaultHisto.clone(
  name = "rechitsize_x",
  title = "X size of RecHit clusters",
  range_min = 0, range_max = 10, range_nbins = 10,
  xlabel = "RecHit X-Size",
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping)
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
    Specification().groupBy(parent(DefaultHisto.defaultGrouping)).save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule).save(),
  )
)

SiPixelPhase1RecHitsClustY = SiPixelPhase1RecHitsClustX.clone(
  name = "rechitsize_y",
  title = "Y size of RecHit clusters",
  xlabel = "RecHit Y-Size"
)

SiPixelPhase1RecHitsErrorX = DefaultHisto.clone(
  name = "rechiterror_x",
  title = "RecHit Error in X-direction",
  range_min = 0, range_max = 0.02, range_nbins = 100,
  xlabel = "X error",
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping)
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
    Specification().groupBy(parent(DefaultHisto.defaultGrouping)).save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule).save(),
  )
)

SiPixelPhase1RecHitsErrorY = SiPixelPhase1RecHitsErrorX.clone(
  name = "rechiterror_y",
  title = "RecHit Error in Y-direction",
  xlabel = "Y error"
)

SiPixelPhase1RecHitsPosition = DefaultHisto.clone(
  name = "rechit_pos",
  title = "Position of RecHits on Module",
  range_min   = -1, range_max   = 1, range_nbins   = 100,
  range_y_min = -4, range_y_max = 4, range_y_nbins = 100,
  xlabel = "x offset",
  ylabel = "y offset",
  dimensions = 2,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping).save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule).save(),
  )
)

SiPixelPhase1RecHitsConf = cms.VPSet(
  SiPixelPhase1RecHitsNRecHits,
  SiPixelPhase1RecHitsClustX,
  SiPixelPhase1RecHitsClustY,
  SiPixelPhase1RecHitsErrorX,
  SiPixelPhase1RecHitsErrorY,
  SiPixelPhase1RecHitsPosition,
)

SiPixelPhase1RecHitsAnalyzer = cms.EDAnalyzer("SiPixelPhase1RecHits",
        src = cms.InputTag("siPixelRecHits"),
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1RecHitsHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1RecHitsConf,
        geometry = SiPixelPhase1Geometry
)
