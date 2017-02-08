import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.SpecificationBuilder_cfi import Specification, parent

SiPixelPhase1Geometry = cms.PSet(
  # Blades are numbered from 1 to n_inner_ring_blades for the inner ring, and 
  # from n_inner_ring_blades+1 to <max_blade> for the outer ring
  n_inner_ring_blades = cms.int32(22), 

  # module geometry. The phase1 detector has only one sort, so this is easy.
  # the values are assumed to be 0-based, unlike most others.
  module_rows = cms.int32(160),
  module_cols = cms.int32(416),
  roc_rows = cms.int32(80),
  roc_cols = cms.int32(52),
  n_rocs = cms.int32(16), # two-row geometry is assumed

  # "time geometry" parameters
  max_lumisection = cms.int32(1000),
  max_bunchcrossing = cms.int32(3600),

  # to select a different cabling map (for pilotBlade)
  CablingMapLabel = cms.string(""),

  # online-secific things
  onlineblock = cms.int32(20),    # #LS after which histograms are reset
  n_onlineblocks = cms.int32(100),  # #blocks to keep for histograms with history

  # other geometry parameters (n_layers, n_ladders per layer, etc.) are inferred.
  # there are lots of geometry assuptions in the code.
)

# the wrapping here is necessary to switch 'enabled' later.
PerModule = cms.PSet(enabled = cms.bool(True)) # normal histos per module
PerLadder = cms.PSet(enabled = cms.bool(True)) # histos per ladder, profiles
PerLayer2D = cms.PSet(enabled = cms.bool(True)) # 2D maps/profiles of layers
PerLayer1D = cms.PSet(enabled = cms.bool(True)) # normal histos per layer
PerLumisection = cms.PSet(enabled = cms.bool(True)) # trend profiles

# Default histogram configuration. This is _not_ used automatically, but you 
# can import and pass this (or clones of it) in the plugin config.
DefaultHisto = cms.PSet(
  # Setting this to False hides all plots of this HistogramManager. It does not even record any data.
  enabled = cms.bool(True),

  # a.k.a. online harvesting. Might be useful in offline for custom harvesting,
  # but the main purpose is online, where this is on by default.
  perLumiHarvesting = cms.bool(False),

  # If False, no histograms are booked for DetIds where any column is undefined.
  # since or-columns are not supported any longer, this has to be False, otherwise
  # you will see a PXBarrel_UNDEFINED with endcap modules and the other way round.
  # It could still be useful for debugging, to see if there is more UNDEFINED 
  # than expected.
  bookUndefined = cms.bool(False),

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
  #     )
  #   )
  # )
  #)
)

DefaultHistoDigiCluster=DefaultHisto.clone()
DefaultHistoDigiCluster.topFolderName= cms.string("PixelPhase1/Phase1_MechanicalView")

DefaultHistoSummary=DefaultHisto.clone()
DefaultHistoSummary.topFolderName= cms.string("PixelPhase1/Summary")

DefaultHistoTrack=DefaultHisto.clone()
DefaultHistoTrack.topFolderName= cms.string("PixelPhase1/Tracks")

# Commonly used specifications. 
StandardSpecifications1D = [
    # The column names are either defined in the GeometryInterface.cc or read from TrackerTopology.
    # Endcap names side by side. The "/" separates columns and also defines how the output folders are nested.

    # per-ladder and profiles
    Specification(PerLadder).groupBy("PXBarrel/Shell/PXLayer/PXLadder")
                            .save()
                            .reduce("MEAN")
                            .groupBy("PXBarrel/Shell/PXLayer", "EXTEND_X")
                            .saveAll(),
    Specification(PerLadder).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade")
                            .save()
                            .reduce("MEAN")
                            .groupBy("PXForward/HalfCylinder/PXDisk/PXRing", "EXTEND_X")
                            .saveAll(),
    Specification(PerLayer1D).groupBy("PXBarrel/Shell/PXLayer").save(),
    Specification(PerLayer1D).groupBy("PXForward/HalfCylinder/PXDisk/PXRing").save(),
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/PXLadder/P1PXModuleName").save(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/P1PXModuleName").save(),
]

StandardSpecificationTrend = [
    Specification().groupBy("PXBarrel/Lumisection")
                   .reduce("MEAN") 
                   .groupBy("PXBarrel", "EXTEND_X")
                   .save(),
    Specification().groupBy("PXForward/Lumisection")
                   .reduce("MEAN") 
                   .groupBy("PXForward", "EXTEND_X")
                   .save(),
]


StandardSpecification2DProfile = [
    Specification(PerLayer2D)
       .groupBy("PXBarrel/PXLayer/signedLadder/signedModule")
       .reduce("MEAN")
       .groupBy("PXBarrel/PXLayer/signedLadder", "EXTEND_X")
       .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXForward/PXDisk/PXBlade/PXPanel")
       .reduce("MEAN")
       .groupBy("PXForward/PXDisk/PXBlade", "EXTEND_X")
       .groupBy("PXForward/PXDisk", "EXTEND_Y")
       .save(),
]

# the same for NDigis and friends. Needed due to technical limitations...
StandardSpecifications1D_Num = [
    Specification(PerLadder).groupBy("PXBarrel/Shell/PXLayer/PXLadder/DetId/Event") 
                            .reduce("COUNT") # per-event counting
                            .groupBy("PXBarrel/Shell/PXLayer/PXLadder").save()
                            .reduce("MEAN")
                            .groupBy("PXBarrel/Shell/PXLayer", "EXTEND_X")
                            .saveAll(),
    Specification(PerModule).groupBy("PXBarrel/Shell/PXLayer/PXLadder/P1PXModuleName/Event")
                            .reduce("COUNT")
                            .groupBy("PXBarrel/Shell/PXLayer/PXLadder/P1PXModuleName")
                            .save(),
    Specification(PerLadder).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/DetId/Event") 
                            .reduce("COUNT") # per-event counting
                            .groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade").save()
                            .reduce("MEAN")
                            .groupBy("PXForward/HalfCylinder/PXDisk/PXRing", "EXTEND_X")
                            .saveAll(),
    Specification(PerModule).groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/P1PXModuleName/Event")
                            .reduce("COUNT")
                            .groupBy("PXForward/HalfCylinder/PXDisk/PXRing/PXBlade/P1PXModuleName")
                            .save(),
]

StandardSpecificationTrend_Num = [
    Specification().groupBy("PXBarrel/Lumisection/PXLayer/Event")
                   .reduce("COUNT")
                   .groupBy("PXBarrel/Lumisection")
                   .reduce("MEAN")
                   .groupBy("PXBarrel", "EXTEND_X")
                   .save(),
    Specification().groupBy("PXForward/Lumisection/PXDisk/Event")
                   .reduce("COUNT")
                   .groupBy("PXForward/Lumisection")
                   .reduce("MEAN")
                   .groupBy("PXForward", "EXTEND_X")
                   .save(),
]


StandardSpecification2DProfile_Num = [
    Specification(PerLayer2D)
       .groupBy("PXBarrel/PXLayer/signedLadder/signedModule" + "/DetId/Event")
       .reduce("COUNT")
       .groupBy("PXBarrel/PXLayer/signedLadder/signedModule")
       .reduce("MEAN") 
       .groupBy("PXBarrel/PXLayer/signedLadder", "EXTEND_X")
       .groupBy("PXBarrel/PXLayer", "EXTEND_Y")
       .save(),
    Specification(PerLayer2D)
       .groupBy("PXForward/PXDisk/PXBlade/PXPanel" + "/DetId/Event")
       .reduce("COUNT")
       .groupBy("PXForward/PXDisk/PXBlade/PXPanel")
       .reduce("MEAN") 
       .groupBy("PXForward/PXDisk/PXBlade", "EXTEND_X")
       .groupBy("PXForward/PXDisk", "EXTEND_Y")
       .save(),
]

# function that makes a VPSet but flattens the argument list if needed
def VPSet(*args):
    l = []
    for a in args:
        if isinstance(a, cms.VPSet) or isinstance(a, Specification):
            e = [a]
        else:
            e = list(a)
        l = l+e
    return cms.VPSet(l)

