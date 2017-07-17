import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

DefaultHistoDebug = DefaultHisto.clone(
  topFolderName = "PixelPhase1/Debug"
)

SiPixelPhase1GeometryDebugDetId = DefaultHistoDebug.clone(
  name = "debug_detid",
  title = "Location of DetIds",
  xlabel = "DetId",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile,
    StandardSpecificationPixelmapProfile,
  )
)

SiPixelPhase1GeometryDebugLadderBlade = DefaultHistoDebug.clone(
  name = "debug_ladderblade",
  title = "Location of Ladders/Blades",
  xlabel = "offline Ladder/Blade #",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile,
    StandardSpecificationPixelmapProfile,
  )
)

SiPixelPhase1GeometryDebugROC = DefaultHistoDebug.clone(
  name = "debug_roc",
  title = "Location of ROCs",
  xlabel = "ROC#",
  dimensions = 1,
  specs = VPSet(
    # TODO: make this per ROC!
    StandardSpecification2DProfile,
    StandardSpecificationPixelmapProfile,
    Specification()
          .groupBy("PXBarrel/PXLayer/PXModuleName/SignedLadderCoord/SignedModuleCoord")
          .groupBy("PXBarrel/PXLayer/PXModuleName/SignedLadderCoord", "EXTEND_X")
          .groupBy("PXBarrel/PXLayer/PXModuleName/", "EXTEND_Y")
          .reduce("MEAN")
          .save(),

  )
)

SiPixelPhase1GeometryDebugFED = DefaultHistoDebug.clone(
  name = "debug_fed",
  title = "Location of FEDs",
  xlabel = "FED#",
  dimensions = 1,
  specs = VPSet(
    StandardSpecification2DProfile,
    StandardSpecificationPixelmapProfile,
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

SiPixelPhase1GeometryDebugHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
    histograms = SiPixelPhase1GeometryDebugConf,
    geometry = SiPixelPhase1Geometry
)
