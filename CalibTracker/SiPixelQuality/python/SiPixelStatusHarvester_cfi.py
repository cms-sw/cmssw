import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
import DQM.SiPixelPhase1Common.TriggerEventFlag_cfi as trigger

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1BadROCp001 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/BadROCp001",
  name = "digi_BadROCp001",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1GoodROCp001 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/GoodROCp001",
  name = "digi_GoodROCp001",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1BadROCp005 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/BadROCp005",
  name = "digi_BadROCp005",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1GoodROCp005 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/GoodROCp005",
  name = "digi_GoodROCp005",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1BadROCp01 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/BadROCp01",
  name = "digi_BadROCp01",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1GoodROCp01 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/GoodROCp01",
  name = "digi_GoodROCp01",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

###########

SiPixelPhase1BadROCp05 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/BadROCp05",
  name = "digi_BadROCp05",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1GoodROCp05 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/GoodROCp05",
  name = "digi_GoodROCp05",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

###########

SiPixelPhase1BadROCp1 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/BadROCp1",
  name = "digi_BadROCp1",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1GoodROCp1 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/GoodROCp1",
  name = "digi_GoodROCp1",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

###########

SiPixelPhase1BadROCp2 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/BadROCp2",
  name = "digi_BadROCp2",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1GoodROCp2 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/GoodROCp2",
  name = "digi_GoodROCp2",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

###########

SiPixelPhase1BadROCp5 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/BadROCp5",
  name = "digi_BadROCp5",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

SiPixelPhase1GoodROCp5 = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityThreshold/GoodROCp5",
  name = "digi_GoodROCp5",
  title = "log_{10}(digi/Nevent)",
  xlabel = "log_{10}(digi/Nevent)",
  range_min = -7,
  range_max = 3,
  range_nbins = 1000,
  ylabel = "N_{lumisection}",
  specs = VPSet(
    Specification().groupBy("PXBarrel").save(),
    Specification().groupBy("PXForward").save(),
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXBarrel/PXLayer/PXModuleName").save(),
    Specification().groupBy("PXForward/PXRing").save(),
    Specification().groupBy("PXForward/PXRing/PXDisk").save()
  )
)

##################

SiPixelPhase1BadROC = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityPCL/BadROC_PCL",
  name = "Dead Channels per ROC",
  title = "Dead Channels per ROC",
  xlabel = "dead channels per ROC",
  range_min = 0, range_max = 1, range_nbins = 100,
  dimensions = 0,
  specs = VPSet(
    Specification(PerLayer2D)
       .groupBy("PXLayer/SignedLadderCoord/SignedModuleCoord")
       .groupBy("PXLayer/SignedLadderCoord", "EXTEND_X")
       .groupBy("PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXRing/SignedBladePanelCoord/SignedDiskCoord")
       .groupBy("PXRing/SignedBladePanelCoord", "EXTEND_X")
       .groupBy("PXRing", "EXTEND_Y")
       .save()
    )
)

SiPixelPhase1PermanentBadROC = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityPCL/BadROC_Permanent",
  name = "Dead Channels per ROC",
  title = "Dead Channels per ROC",
  xlabel = "dead channels per ROC",
  range_min = 0, range_max = 1, range_nbins = 100,
  dimensions = 0,
  specs = VPSet(
    Specification(PerLayer2D)
       .groupBy("PXLayer/SignedLadderCoord/SignedModuleCoord")
       .groupBy("PXLayer/SignedLadderCoord", "EXTEND_X")
       .groupBy("PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXRing/SignedBladePanelCoord/SignedDiskCoord")
       .groupBy("PXRing/SignedBladePanelCoord", "EXTEND_X")
       .groupBy("PXRing", "EXTEND_Y")
       .save()
    )
)

SiPixelPhase1FEDerrorROC = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityPCL/BadROC_FEDerror",
  name = "Dead Channels per ROC",
  title = "Dead Channels per ROC",
  xlabel = "dead channels per ROC",
  range_min = 0, range_max = 1, range_nbins = 100,
  dimensions = 0,
  specs = VPSet(
    Specification(PerLayer2D)
       .groupBy("PXLayer/SignedLadderCoord/SignedModuleCoord")
       .groupBy("PXLayer/SignedLadderCoord", "EXTEND_X")
       .groupBy("PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXRing/SignedBladePanelCoord/SignedDiskCoord")
       .groupBy("PXRing/SignedBladePanelCoord", "EXTEND_X")
       .groupBy("PXRing", "EXTEND_Y")
       .save()
    )
)

SiPixelPhase1StuckTBMROC = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityPCL/BadROC_StuckTBM",
  name = "Dead Channels per ROC",
  title = "Dead Channels per ROC",
  xlabel = "dead channels per ROC",
  range_min = 0, range_max = 1, range_nbins = 100,
  dimensions = 0,
  specs = VPSet(
    Specification(PerLayer2D)
       .groupBy("PXLayer/SignedLadderCoord/SignedModuleCoord")
       .groupBy("PXLayer/SignedLadderCoord", "EXTEND_X")
       .groupBy("PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXRing/SignedBladePanelCoord/SignedDiskCoord")
       .groupBy("PXRing/SignedBladePanelCoord", "EXTEND_X")
       .groupBy("PXRing", "EXTEND_Y")
       .save()
    )
)

SiPixelPhase1OtherBadROC = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityPCL/BadROC_Other",
  name = "Dead Channels per ROC",
  title = "Dead Channels per ROC",
  xlabel = "dead channels per ROC",
  range_min = 0, range_max = 1, range_nbins = 100,
  dimensions = 0,
  specs = VPSet(
    Specification(PerLayer2D)
       .groupBy("PXLayer/SignedLadderCoord/SignedModuleCoord")
       .groupBy("PXLayer/SignedLadderCoord", "EXTEND_X")
       .groupBy("PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXRing/SignedBladePanelCoord/SignedDiskCoord")
       .groupBy("PXRing/SignedBladePanelCoord", "EXTEND_X")
       .groupBy("PXRing", "EXTEND_Y")
       .save()
    )
)

SiPixelPhase1PromptBadROC = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/SiPixelQualityPCL/BadROC_Prompt",
  name = "Dead Channels per ROC",
  title = "Dead Channels per ROC",
  xlabel = "dead channels per ROC",
  range_min = 0, range_max = 1, range_nbins = 100,
  dimensions = 0,
  specs = VPSet(
    Specification(PerLayer2D)
       .groupBy("PXLayer/SignedLadderCoord/SignedModuleCoord")
       .groupBy("PXLayer/SignedLadderCoord", "EXTEND_X")
       .groupBy("PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXRing/SignedBladePanelCoord/SignedDiskCoord")
       .groupBy("PXRing/SignedBladePanelCoord", "EXTEND_X")
       .groupBy("PXRing", "EXTEND_Y")
       .save()
    )
)

# This has to match the order of the names in the C++ enum.
SiPixelPhase1BadROCConf = cms.VPSet(
SiPixelPhase1BadROCp001,
SiPixelPhase1GoodROCp001,
SiPixelPhase1BadROCp005,
SiPixelPhase1GoodROCp005,
SiPixelPhase1BadROCp01,
SiPixelPhase1GoodROCp01,
SiPixelPhase1BadROCp05,
SiPixelPhase1GoodROCp05,
SiPixelPhase1BadROCp1,
SiPixelPhase1GoodROCp1,
SiPixelPhase1BadROCp2,
SiPixelPhase1GoodROCp2,
SiPixelPhase1BadROCp5,
SiPixelPhase1GoodROCp5,
SiPixelPhase1BadROC,
SiPixelPhase1PermanentBadROC,
SiPixelPhase1FEDerrorROC,
SiPixelPhase1StuckTBMROC, 
SiPixelPhase1OtherBadROC,
SiPixelPhase1PromptBadROC
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
siPixelStatusHarvester = DQMEDAnalyzer("SiPixelStatusHarvester",
    histograms = SiPixelPhase1BadROCConf,
    geometry = SiPixelPhase1Geometry,
    #triggerflags = trigger.SiPixelPhase1Triggers SiPixelQuality ALCARECO doesn't contain any trigger infor
    SiPixelStatusManagerParameters = cms.PSet(
        thresholdL1  = cms.untracked.double(0.1),
        thresholdL2  = cms.untracked.double(0.2),
        thresholdL3  = cms.untracked.double(0.2),
        thresholdL4  = cms.untracked.double(0.2),
        thresholdRNG1  = cms.untracked.double(0.1),
        thresholdRNG2  = cms.untracked.double(0.1),
        outputBase = cms.untracked.string("runbased"), #nLumibased #runbased #dynamicLumibased
        aveDigiOcc = cms.untracked.int32(20000),
        resetEveryNLumi = cms.untracked.int32(1),
        moduleName = cms.untracked.string("siPixelStatusProducer"),
        label      = cms.untracked.string("siPixelStatus"),
    ),
    debug = cms.untracked.bool(False),
    recordName   = cms.untracked.string("SiPixelQualityFromDbRcd")

)

siPixelPhase1DQMHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1BadROCConf,
        geometry = SiPixelPhase1Geometry
)

