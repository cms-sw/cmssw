import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.SpecificationBuilder_cfi import Specification, parent

SiPixelPhase1Geometry = cms.PSet(
  # No options atm.
)

# the wrapping here is necessary to switch 'enabled' later.
PerModule = cms.PSet(enabled = cms.bool(True))

# Default histogram configuration. This is _not_ used automatically, but you 
# can import and pass this (or clones of it) in the plugin config.
DefaultHisto = cms.PSet(
  # Setting this to False hides all plots of this HistogramManager. It does not even record any data.
  enabled = cms.bool(True),
  # If False, no histograms are booked for DetIds where any column is undefined.
  bookUndefined = cms.bool(True),
  # where the plots should go.
  topFolderName = cms.string("PixelPhase1"),

  # Histogram parameters
  name = cms.string("unnamed"),
  title = cms.string("Histogram of Something"),
  xlabel = cms.string("something"),
  ylabel = cms.string("count"),
  dimensions = cms.int32(1),
  range_min = cms.double(0),
  range_max = cms.double(100), 
  range_nbins = cms.int32(100),
  range_y_min = cms.double(0),
  range_y_max = cms.double(100), 
  range_y_nbins = cms.int32(100),

  # This grouping should be used as a default (explicitly in the Plugin config). It should be era-dependent.
  # The column names are either defined in the GeometryInterface.cc or read from TrackerTopology.
  # The "|" means "try the first, if not present try the second", it should be used to have Barrel- and 
  # Endcap names side by side. The "/" separates columns and also defines how the output folders are nested.
  defaultGrouping  = cms.string("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade"),
  defaultPerModule = cms.string("PXBarrel|PXForward/Shell|HalfCylinder/PXLayer|PXDisk/PXRing|/PXLadder|PXBlade/PXBModule|PXPanel"),

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

