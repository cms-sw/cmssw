import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackResidualsResidualsX = DefaultHistoTrack.clone(
  name = "residual_x",
  title = "Track Residuals X",
  range_min = -0.5, range_max = 0.5, range_nbins = 200,
  xlabel = "(x_rec - x_pred) [cm]",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile,
    StandardSpecifications1D
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
        VertexCut                  = cms.bool(True), # Will not apply the vertex cuts on cosmics
        histograms = SiPixelPhase1TrackResidualsConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackResidualsHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1TrackResidualsConf,
        geometry = SiPixelPhase1Geometry
)
