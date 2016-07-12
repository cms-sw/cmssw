import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackResidualsResidualsX = DefaultHisto.clone(
  name = "residual_x",
  title = "Track Residuals X",
  range_min = -0.05, range_max = 0.05, range_nbins = 100,
  xlabel = "(x_rec - x_pred) [cm]",
  dimensions = 1,
  specs = cms.VPSet(
    Specification().groupBy(DefaultHisto.defaultGrouping) 
                   .save()
                   .reduce("MEAN")
                   .groupBy(parent(DefaultHisto.defaultGrouping), "EXTEND_X")
                   .saveAll(),
    Specification().groupBy(parent(DefaultHisto.defaultGrouping))
                   .save(),
    Specification(PerModule).groupBy(DefaultHisto.defaultPerModule)
                            .save()
  )
)

SiPixelPhase1TrackResidualsResidualsY = SiPixelPhase1TrackResidualsResidualsX.clone(
  name = "residual_y",
  title = "Track Residuals Y",
  xlabel = "(y_rec - y_pred) [cm]",
)

SiPixelPhase1TrackResidualsConf = cms.VPSet(
  SiPixelPhase1TrackResidualsResidualsX,
  SiPixelPhase1TrackResidualsResidualsY
)

SiPixelPhase1TrackResidualsAnalyzer = cms.EDAnalyzer("SiPixelPhase1TrackResiduals",
        trajectoryInput = cms.string("generalTracks"),
        Tracks        = cms.InputTag("generalTracks"),
        histograms = SiPixelPhase1TrackResidualsConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackResidualsHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackResidualsConf,
        geometry = SiPixelPhase1Geometry
)
