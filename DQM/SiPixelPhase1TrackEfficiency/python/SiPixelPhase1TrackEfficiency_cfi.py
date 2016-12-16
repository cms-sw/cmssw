import FWCore.ParameterSet.Config as cms
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1TrackEfficiencyValid = DefaultHistoTrack.clone(
  name = "valid",
  title = "Valid Hits",
  dimensions = 0,
  specs = VPSet(
    # custom() is called here after every save to export the histos for the
    # efficiency harvesting. The parameter is just a tag that we don't confuse 
    # the histos of different specs.
    Specification().groupBy("PXForward/PXDisk/PXBlade/PXPanel")
                   .groupBy("PXForward/PXDisk/PXBlade", "EXTEND_X")
                   .groupBy("PXForward/PXDisk", "EXTEND_Y")
                   .save(),
    Specification().groupBy("PXBarrel/PXLayer/signedLadder/signedModule")
                   .groupBy("PXBarrel/PXLayer/signedLadder", "EXTEND_X")
                   .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
                   .save()
                   .custom("signedmodule_barrel"),
    Specification().groupBy("PXBarrel/PXLayer/ROC")
                   .groupBy("PXBarrel/PXLayer", "EXTEND_X")
                   .save()
                   .custom("roc_barrel"),
    
    StandardSpecifications1D_Num
  )
)

SiPixelPhase1TrackEfficiencyMissing = SiPixelPhase1TrackEfficiencyValid.clone(
  name = "missing",
  title = "Missing Hits",
)

SiPixelPhase1TrackEfficiencyEfficiency = SiPixelPhase1TrackEfficiencyValid.clone(
  name = "hitefficiency",
  title = "Hit Efficiency",
  # most stuff done in custom() step, carried over from valid/missing histos
  # the custom() step looks for matching valid/mmissing histos and fills the
  # efficiency plots if data is available. So all should use the same specs.
)

SiPixelPhase1TrackEfficiencyEfficiencyVertices= DefaultHistoTrack.clone(
    name = "num_vertices",
    title = "PrimaryVertices",
    xlabel= "# Vertices",
    dimensions = 1,
    range_min = -0.5,
    range_max = 100.5, 
    range_nbins =101,
    specs = VPSet(
        Specification().groupBy("")
                   .save(),
        Specification().groupBy("/Lumisection")
                   .reduce("MEAN")
                   .groupBy("","EXTEND_X")
                   .save()
   )
 )



SiPixelPhase1TrackEfficiencyConf = cms.VPSet(
  SiPixelPhase1TrackEfficiencyValid,
  SiPixelPhase1TrackEfficiencyMissing,
  SiPixelPhase1TrackEfficiencyEfficiency,
  SiPixelPhase1TrackEfficiencyEfficiencyVertices
)


SiPixelPhase1TrackEfficiencyAnalyzer = cms.EDAnalyzer("SiPixelPhase1TrackEfficiency",
        clusters = cms.InputTag("siPixelClusters"),
        trajectories = cms.InputTag("generalTracks"),
        primaryvertices = cms.InputTag("offlinePrimaryVertices"),
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)

SiPixelPhase1TrackEfficiencyHarvester = cms.EDAnalyzer("SiPixelPhase1TrackEfficiencyHarvester",
        histograms = SiPixelPhase1TrackEfficiencyConf,
        geometry = SiPixelPhase1Geometry
)
