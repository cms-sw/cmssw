import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
import DQM.SiPixelPhase1Common.TriggerEventFlag_cfi as trigger

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DeadChannelsPerFED = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/FED",
  name = "Dead Channels",
  title = "Dead Channels",
  xlabel = "dead channels",
  range_min = 0, range_max = 1000, range_nbins = 100,
  dimensions = 0,
  specs = VPSet(
    Specification().groupBy("FED/Event")
                   .reduce("COUNT")
                   .groupBy("FED")
                   .reduce("MEAN")
                   .groupBy("","EXTEND_X")
                   .save(), #average dead channels per event and FED                  
    Specification().groupBy("FED/Event")
                   .reduce("COUNT")
                   .groupBy("FED/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("FED","EXTEND_X")
                   .groupBy("","EXTEND_Y")
                   .save(), #average dead channels per event and FED per LumiBlock 
    Specification().groupBy("PXAll/Event")
                   .reduce("COUNT")
                   .groupBy("LumiBlock") #average number of dead channels per Lumisection
                   .reduce("MEAN")
                   .groupBy("", "EXTEND_X")
                   .save(),
    Specification().groupBy("PXBarrel/Event")
                   .reduce("COUNT")
                   .groupBy("PXBarrel")
                   .save(),
    Specification().groupBy("PXForward/Event")
                   .reduce("COUNT")
                   .groupBy("PXForward")
                   .save(),
    Specification().groupBy("PXBarrel/Event")
                   .reduce("COUNT")
                   .groupBy("PXBarrel/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("PXBarrel","EXTEND_X")
                   .save(),
    Specification().groupBy("PXForward/Event")
                   .reduce("COUNT")
                   .groupBy("PXForward/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("PXForward","EXTEND_X")
                   .save(),
    Specification().groupBy("PXBarrel/PXLayer/Event")
                   .reduce("COUNT")
                   .groupBy("PXBarrel/PXLayer/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("PXBarrel/PXLayer","EXTEND_X")
                   .groupBy("PXBarrel", "EXTEND_Y")
                   .save(),
    Specification().groupBy("PXForward/PXDisk/Event")
                   .reduce("COUNT")
                   .groupBy("PXForward/PXDisk/LumiBlock")
                   .reduce("MEAN")
                   .groupBy("PXForward/PXDisk","EXTEND_X")
                   .groupBy("PXForward", "EXTEND_Y")
                   .save(),
    Specification().groupBy("FED/LinkInFed/Event")
                   .reduce("COUNT")
                   .groupBy("FED/LinkInFed")
                   .reduce("MEAN")
                   .groupBy("FED","EXTEND_X")
                   .groupBy("","EXTEND_Y")
                   .save()
    )
)


SiPixelPhase1DeadChannelsPerROC = DefaultHisto.clone(
topFolderName = DefaultHisto.topFolderName.value() +"/FED",
  name = "Dead Channels per ROC",
  title = "Dead Channels per ROC",
  xlabel = "dead channels per ROC",
  range_min = 0, range_max = 1000, range_nbins = 100,
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
SiPixelPhase1DeadFEDChannelsConf = cms.VPSet(
SiPixelPhase1DeadChannelsPerFED ,
SiPixelPhase1DeadChannelsPerROC 
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelPhase1DeadFEDChannelsAnalyzer = DQMEDAnalyzer('SiPixelPhase1DeadFEDChannels',
        histograms = SiPixelPhase1DeadFEDChannelsConf,
        geometry = SiPixelPhase1Geometry,
        triggerflags = trigger.SiPixelPhase1Triggers
)

SiPixelPhase1DeadFEDChannelsHarvester = DQMEDHarvester("SiPixelPhase1Harvester",
        histograms = SiPixelPhase1DeadFEDChannelsConf,
        geometry = SiPixelPhase1Geometry
)
