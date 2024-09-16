import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.HistogramManager_cfi import *
OverlayCurvesForTiming.enabled = False #switch to overlay digi/clusters curves for timing scan 
#PerModule.enabled              = False 
#PerLadder.enabled              = False
#PerLayer2D.enabled             = True # 2D maps/profiles of layers
#PerLayer1D.enabled             = True # normal histos per layer

hltSiPixelPhase1Geometry = SiPixelPhase1Geometry.clone(
    max_lumisection   = 2500,
    max_bunchcrossing = 3600,
    # online-specific things
    onlineblock    =  20, # #LS after which histograms are reset
    n_onlineblocks = 100 # #blocks to keep for histograms with history
)
hltDefaultHistoDigiCluster = DefaultHistoDigiCluster.clone(
    topFolderName = "HLT/Pixel"
)
hltDefaultHistoReadout = DefaultHistoReadout.clone(
    topFolderName = "HLT/Pixel"
)
hltDefaultHistoTrack = DefaultHistoTrack.clone(
    topFolderName= "HLT/Pixel/TrackClusters"
)
hltDefaultHistoTrackResiduals = DefaultHistoTrack.clone(
    topFolderName= "HLT/Pixel/Tracks"
)

hltStandardSpecificationPixelmapProfile = [#produces pixel map with the mean (TProfile)
    Specification(PerLayer2D)
       .groupBy("PXBarrel/PXLayer/SignedLadderCoord/SignedModuleCoord")
       .groupBy("PXBarrel/PXLayer/SignedLadderCoord", "EXTEND_X")
       .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
       .reduce("MEAN")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXForward/PXRing/SignedBladePanelCoord/SignedDiskCoord")
       .groupBy("PXForward/PXRing/SignedBladePanelCoord", "EXTEND_X")
       .groupBy("PXForward/PXRing", "EXTEND_Y")
       .reduce("MEAN")
       .save(),
]

hltStandardSpecifications1D = [
    Specification().groupBy("PXBarrel/PXLayer").save(),
    Specification().groupBy("PXForward/PXDisk").save(), # 91x nbins=100, xmin=0, xmax=3000,
]

hltStandardSpecifications1D_Num = [
    Specification().groupBy("PXBarrel/PXLayer/Event") #this will produce inclusive counts per Layer/Disk
    .reduce("COUNT")    
    .groupBy("PXBarrel/PXLayer")
#    .groupBy("PXBarrel")
    .save(),
    Specification().groupBy("PXForward/PXDisk/Event")
    .reduce("COUNT")    
    .groupBy("PXForward/PXDisk/")
#    .groupBy("PXForward")
    .save()
]
