import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1GeometryDebugDetId = DefaultHisto.clone(
  name = "debug_detid",
  title = "Location of DetIds",
  xlabel = "DetId",
  dimensions = 1,
  specs = cms.VPSet(
    StandardSpecification2DProfile
  )
)

SiPixelPhase1GeometryDebugLadderBlade = DefaultHisto.clone(
  name = "debug_ladderblade",
  title = "Location of Ladders/Blades",
  xlabel = "offline Ladder/Blade #",
  dimensions = 1,
  specs = cms.VPSet(
    StandardSpecification2DProfile
  )
)

SiPixelPhase1GeometryDebugROC = DefaultHisto.clone(
  name = "debug_roc",
  title = "Location of ROCs",
  xlabel = "ROC#",
  dimensions = 1,
  specs = cms.VPSet(
    # TODO: make this per ROC!
    StandardSpecification2DProfile
  )
)

SiPixelPhase1GeometryDebugFED = DefaultHisto.clone(
  name = "debug_fed",
  title = "Location of FEDs",
  xlabel = "FED#",
  dimensions = 1,
  specs = cms.VPSet(
    StandardSpecification2DProfile
  )
)

SiPixelPhase1GeometryDebugConf = cms.VPSet(
  SiPixelPhase1GeometryDebugDetId,
  SiPixelPhase1GeometryDebugLadderBlade,
  SiPixelPhase1GeometryDebugROC,
  SiPixelPhase1GeometryDebugFED,
)

SiPixelPhase1GeometryDebugAnalyzer = cms.EDAnalyzer("SiPixelPhase1GeometryDebug",
    histograms = SiPixelPhase1GeometryDebugConf,
    geometry = SiPixelPhase1Geometry
)

SiPixelPhase1GeometryDebugHarvester = cms.EDAnalyzer("SiPixelPhase1Harvester",
    histograms = SiPixelPhase1GeometryDebugConf,
    geometry = SiPixelPhase1Geometry
)
