import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_HistogramManager_cfi import *

# order is important and it should follow ordering in hltSiPixelPhase1ClustersConf VPSet
hltSiPixelPhase1TrackResidualsResidualsX = hltDefaultHistoTrackResiduals.clone(
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

hltSiPixelPhase1TrackResidualsResidualsY = hltSiPixelPhase1TrackResidualsResidualsX.clone(
  name = "residual_y",
  title = "Track Residuals Y",
  xlabel = "(y_rec - y_pred) [cm]",
)

hltSiPixelPhase1TrackResidualsResOnEdgeX = hltDefaultHistoTrackResiduals.clone(
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

hltSiPixelPhase1TrackResidualsResOnEdgeY = hltSiPixelPhase1TrackResidualsResOnEdgeX.clone(
  name = "residual_OnEdge_y",
  title = "Track Residuals Y (OnEdge Clusters)",
  xlabel = "(y_rec - y_pred) [cm]",
)


hltSiPixelPhase1TrackResidualsResOtherBadX = hltDefaultHistoTrackResiduals.clone(
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

hltSiPixelPhase1TrackResidualsResOtherBadY = hltSiPixelPhase1TrackResidualsResOtherBadX.clone(
  name = "residual_OtherBad_y",
  title = "Track Residuals Y (OtherBad Clusters)",
  xlabel = "(y_rec - y_pred) [cm]",
)


hltSiPixelPhase1TrackNormResX = hltDefaultHistoTrackResiduals.clone(
  topFolderName= cms.string("HLT/Pixel/Tracks/ResidualsExtra"),
  name = "NormRes_x",
  title = "Normalized Residuals X",
  range_min = -5, range_max = 5, range_nbins = 100,
  xlabel = "(x_rec - x_pred)/x_err",
  dimensions = 1,
  specs = VPSet(
    Specification().groupBy("PXBarrel/PXLayer").saveAll(),
    Specification().groupBy("PXForward/PXDisk").saveAll(),
    Specification(PerLayer1D).groupBy("PXBarrel/Shell/PXLayer").save(),
    Specification(PerLayer1D).groupBy("PXForward/HalfCylinder/PXRing/PXDisk").save()
  )
)

hltSiPixelPhase1TrackNormResY = hltSiPixelPhase1TrackNormResX.clone(
  name = "NormRes_y",
  title = "Normalized Residuals Y",
  range_min = -5, range_max = 5, range_nbins = 100,
  xlabel = "(y_rec - y_pred)/y_err",
)

hltSiPixelPhase1TrackDRnRX = hltDefaultHistoTrackResiduals.clone(
  topFolderName= cms.string("HLT/Pixel/Tracks/ResidualsExtra"),
  name = "DRnR_x",
  title = "Distribution of RMS of Normalized Residuals X",
  range_min = -5, range_max = 5, range_nbins = 100,
  xlabel = "#sigma_{(x_rec - x_pred)/x_err}",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile
  )
)

hltSiPixelPhase1TrackDRnRY = hltSiPixelPhase1TrackDRnRX.clone(
  name = "DRnR_y",
  title = "Distribution of RMS of Normalized Residuals Y",
  range_min = -5, range_max = 5, range_nbins = 100,
  xlabel = "#sigma_{(y_rec - y_pred)/y_err}",
)

hltSiPixelPhase1TrackResidualsConf = cms.VPSet(
  hltSiPixelPhase1TrackResidualsResidualsX,
  hltSiPixelPhase1TrackResidualsResidualsY,
  hltSiPixelPhase1TrackResidualsResOnEdgeX,
  hltSiPixelPhase1TrackResidualsResOnEdgeY,
  hltSiPixelPhase1TrackResidualsResOtherBadX,
  hltSiPixelPhase1TrackResidualsResOtherBadY,
  hltSiPixelPhase1TrackNormResX,
  hltSiPixelPhase1TrackNormResY,
  hltSiPixelPhase1TrackDRnRX,
  hltSiPixelPhase1TrackDRnRY
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hltSiPixelPhase1TrackResidualsAnalyzer = DQMEDAnalyzer('SiPixelPhase1TrackResiduals',
        trajectoryInput = cms.string("hltTrackRefitterForPixelDQM"),
        Tracks        = cms.InputTag("hltTrackRefitterForPixelDQM"),
        vertices = cms.InputTag("hltPixelVertices"),
        histograms = hltSiPixelPhase1TrackResidualsConf,
        geometry = hltSiPixelPhase1Geometry,
        VertexCut = cms.untracked.bool(True)
)

from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
pp_on_PbPb_run3.toModify(hltSiPixelPhase1TrackResidualsAnalyzer,
                         vertices = 'hltPixelVerticesPPOnAA')

hltSiPixelPhase1TrackResidualsHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = hltSiPixelPhase1TrackResidualsConf,
        geometry = hltSiPixelPhase1Geometry
)

hltSiPixelPhase1ResidualsExtra = DQMEDHarvester("SiPixelPhase1ResidualsExtra",
    TopFolderName = cms.string('HLT/Pixel/Tracks/ResidualsExtra'),
    InputFolderName = cms.string('HLT/Pixel/Tracks'),
    MinHits = cms.int32(30)
)
