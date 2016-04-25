import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.SpecificationBuilder_cfi import Specification, parent

SiPixelPhase1Geometry = cms.PSet(
  # No options atm.
)

# Default histogram configuration. This is _not_ used automatically, but you 
# can import and pass this (or clones of it) in the plugin config.
DefaultHisto = cms.PSet(
  # Setting this to False hides all plots of this HistogramManager. It does not even record any data.
  enabled = cms.bool(True),
  # If False, no histograsm are booked for DetIds where any column is undefined.
  # This is important to avoid booking lots of unused histograms for SiStrip IDs.
  bookUndefined = cms.bool(False),
  # where the plots should go.
  topFolderName = cms.string("PixelPhase1"),

  # Histogram parameters
  name = cms.string("unnamed"),
  title = cms.string("Histogram of Something"),
  xlabel = cms.string("something"),
  ylabel = cms.string("count"),
  range_min = cms.double(0),
  range_max = cms.double(100), 
  range_nbins = cms.int32(100),
  dimensions = cms.int32(1),

  # This grouping should be used as a default (explicitly in the Plugin config). It should be era-dependent.
  # The column names are either defined in the GeometryInterface.cc or read from TrackerTopology.
  # The "|" means "try the first, if not present try the second", it should be used to have Barrel- and 
  # Endcap names side by side. The "/" separates columns and also defines how the output folders are nested.
  defaultGrouping = cms.string("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXLadder|PXBlade"),

  # This structure is output by the SpecficationBuilder.
  specs = cms.VPSet()
  #  cms.PSet(spec = 
  #    cms.VPset(
  #      cms.PSet(
  #        type = GROUPBY, 
  #        stage = FIRST,
  #        columns = cms.vstring("P1PXBBarrel|P1PXECEndcap", "DetId"),
  #        arg = cms.string("")
  #      ),
  #     cms.PSet(
  #       type = SAVE,
  #       stage = STAGE1,
  #       columns = cms.vstring(),
  #       arg = cms.string("")
  #	)
  #   )
  # )
  #)
)

