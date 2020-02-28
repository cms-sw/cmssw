import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *
import DQM.SiPixelPhase1Common.TriggerEventFlag_cfi as trigger

SiPixelPhase1TrackResidualsResidualsX = DefaultHistoTrack.clone(
  name = "residual_x",
  title = "Track Residuals X",
  range_min = -0.1, range_max = 0.1, range_nbins = 100,
  xlabel = "(x_rec - x_pred) [cm]",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile,
    Specification().groupBy("PXBarrel/PXLayer").saveAll(),
    Specification().groupBy("PXForward/PXDisk").saveAll(),
    
    Specification().groupBy("PXBarrel/PXLayer/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("PXBarrel/PXLayer", "EXTEND_X")
                   .save(),

    Specification().groupBy("PXForward/PXDisk/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("PXForward/PXDisk", "EXTEND_X")
                   .save(),

    Specification(PerLayer1D).groupBy("PXBarrel/Shell/PXLayer").save(),
    Specification(PerLayer1D).groupBy("PXForward/HalfCylinder/PXRing/PXDisk").save()
  )
)

SiPixelPhase1TrackResidualsResidualsY = SiPixelPhase1TrackResidualsResidualsX.clone(
  name = "residual_y",
  title = "Track Residuals Y",
  xlabel = "(y_rec - y_pred) [cm]",
)

SiPixelPhase1TrackResidualsResOnEdgeX = DefaultHistoTrack.clone(
  name = "residual_OnEdge_x",
  title = "Track Residuals X (OnEdge Clusters)",
  range_min = -0.1, range_max = 0.1, range_nbins = 100,
  xlabel = "(x_rec - x_pred) [cm]",
  dimensions = 1,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").saveAll(),
    Specification().groupBy("PXForward/PXDisk").saveAll(),
    Specification(PerLayer1D).groupBy("PXBarrel/Shell/PXLayer").save(),
    Specification(PerLayer1D).groupBy("PXForward/HalfCylinder/PXRing/PXDisk").save()
  )
)

SiPixelPhase1TrackResidualsResOnEdgeY = SiPixelPhase1TrackResidualsResOnEdgeX.clone(
  name = "residual_OnEdge_y",
  title = "Track Residuals Y (OnEdge Clusters)",
  xlabel = "(y_rec - y_pred) [cm]",
)


SiPixelPhase1TrackResidualsResOtherBadX = DefaultHistoTrack.clone(
  name = "residual_OtherBad_x",
  title = "Track Residuals X (OtherBad Clusters)",
  range_min = -0.1, range_max = 0.1, range_nbins = 100,
  xlabel = "(x_rec - x_pred) [cm]",
  dimensions = 1,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").saveAll(),
    Specification().groupBy("PXForward/PXDisk").saveAll(),
    Specification(PerLayer1D).groupBy("PXBarrel/Shell/PXLayer").save(),
    Specification(PerLayer1D).groupBy("PXForward/HalfCylinder/PXRing/PXDisk").save()
  )
)

SiPixelPhase1TrackResidualsResOtherBadY = SiPixelPhase1TrackResidualsResOtherBadX.clone(
  name = "residual_OtherBad_y",
  title = "Track Residuals Y (OtherBad Clusters)",
  xlabel = "(y_rec - y_pred) [cm]",
)


SiPixelPhase1TrackResidualsConf = cms.VPSet(
  SiPixelPhase1TrackResidualsResidualsX,
  SiPixelPhase1TrackResidualsResidualsY,
  SiPixelPhase1TrackResidualsResOnEdgeX,
  SiPixelPhase1TrackResidualsResOnEdgeY,
  SiPixelPhase1TrackResidualsResOtherBadX,
  SiPixelPhase1TrackResidualsResOtherBadY
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelPhase1TrackResidualsAnalyzer = DQMEDAnalyzer('SiPixelPhase1TrackResiduals',
        trajectoryInput = cms.string("generalTracks"),
        Tracks        = cms.InputTag("generalTracks"),
        vertices = cms.InputTag("offlinePrimaryVertices"),
        histograms = SiPixelPhase1TrackResidualsConf,
        geometry = SiPixelPhase1Geometry,
        triggerflags = trigger.SiPixelPhase1Triggers
)

SiPixelPhase1TrackResidualsHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackResidualsConf,
        geometry = SiPixelPhase1Geometry
)
