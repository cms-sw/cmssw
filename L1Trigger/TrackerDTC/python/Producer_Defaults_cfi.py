import FWCore.ParameterSet.Config as cms

TrackerDTCProducer_params = cms.PSet (

  #=== ED parameter

  ParamsED = cms.PSet (
    InputTagTTStubDetSetVec       = cms.InputTag  ( "TTStubsFromPhase2TrackerDigis", "StubAccepted" ),   # 
    ProductBranch                 = cms.string    ( "StubAccepted" ),                                    #
    InputTagMagneticField         = cms.ESInputTag( "VolumeBasedMagneticFieldESProducer", "" ),          #
    InputTagTrackerGeometry       = cms.ESInputTag( "trackerGeometry", "" ),                             #
    InputTagTrackerTopology       = cms.ESInputTag( "trackerTopology", "" ),                             #
    InputTagCablingMap            = cms.ESInputTag( "GlobalTag", "" ),                                   #
    InputTagTTStubAlgorithm       = cms.ESInputTag( "TTStubAlgorithm_official_Phase2TrackerDigi_", ""),  #
    InputTagGeometryConfiguration = cms.ESInputTag( "XMLIdealGeometryESSource", ""),                     #
    SupportedTrackerXMLPSet       = cms.string    ( "geomXMLFiles"),                                     #
    SupportedTrackerXMLPath       = cms.string    ( "Geometry/TrackerCommonData/data/PhaseII/"),         #
    SupportedTrackerXMLFile       = cms.string    ( "tracker.xml"),                                      #
    SupportedTrackerXMLVersions   = cms.vstring   ( "TiltedTracker613", "TiltedTracker613_MB_2019_04", "OuterTracker616_2020_04" ), #
    DataFormat                    = cms.string    ( "Hybrid" ),                                          # hybrid and tmtt format supported
    OffsetDetIdDSV                = cms.int32     (  1 ),                                                # tk layout det id minus DetSetVec->detId
    OffsetDetIdTP                 = cms.int32     ( -1 ),                                                # tk layout det id minus TrackerTopology lower det id
    OffsetLayerDisks              = cms.int32     ( 10 ),                                                # offset in layer ids between barrel layer and endcap disks
    OffsetLayerId                 = cms.int32     (  1 ),                                                # offset between 0 and smallest layer id (barrel layer 1)
    CheckHistory                  = cms.bool      ( False ),                                              #
    ProcessName                   = cms.string    ( "HLT" ),                                             #
    ProductLabel                  = cms.string    ( "XMLIdealGeometryESSource" )                         #
  ),

  #=== router parameter

  ParamsRouter = cms.PSet (
    EnableTruncation = cms.bool  ( False  ), # enables emulation of truncation
    FreqDTC          = cms.double( 360.  ), # Frequency in MHz, has to be integer multiple of FreqLHC
    TMP_TFP          = cms.int32 (  18   ), # time multiplexed period of track finding processor
    NumFramesInfra   = cms.int32 (   6   ), # needed gap between events of emp-infrastructure firmware
    NumRoutingBlocks = cms.int32 (   2   ), # number of systiloic arrays in stub router firmware
    SizeStack        = cms.int32 (  64   )  # fifo depth in stub router firmware
  ),

  #=== converter parameter

  ParamsConverter = cms.PSet (
    WidthRowLUT    = cms.int32 ( 4 ), # number of row bits used in look up table
    WidthQoverPt   = cms.int32 ( 9 )  # number of bits used for stub qOverPt. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
  ),

  #=== Tracker parameter

  ParamsTracker           = cms.PSet (
    NumRegions            = cms.int32 (  9      ), # number of phi slices the outer tracker readout is organized in
    NumOverlappingRegions = cms.int32 (  2      ), # number of regions a reconstructable particles may cross
    NumDTCsPerRegion      = cms.int32 ( 24      ), # number of DTC boards used to readout a detector region
    NumModulesPerDTC      = cms.int32 ( 72      ), # max number of sensor modules connected to one DTC board
    TMP_FE                = cms.int32 (  8      ), # number of events collected in front-end
    WidthBend             = cms.int32 (  6      ), # number of bits used for internal stub bend
    WidthCol              = cms.int32 (  5      ), # number of bits used for internal stub column
    WidthRow              = cms.int32 ( 11      ), # number of bits used for internal stub row
    BaseBend              = cms.double(   .25   ), # precision of internal stub bend in pitch units
    BaseCol               = cms.double(  1.     ), # precision of internal stub column in pitch units
    BaseRow               = cms.double(   .5    ), # precision of internal stub row in pitch units
    BendCut               = cms.double(  1.3125 ), # used stub bend uncertainty in pitch units
    FreqLHC               = cms.double( 40.     )  # LHC bunch crossing rate in MHz
  ),

  #=== f/w constants

  ParamsFW = cms.PSet (
    SpeedOfLight = cms.double( 2.99792458       ), # in e8 m/s
    BField       = cms.double( 3.81120228767395 ), # in T
    BFieldError  = cms.double(   1.e-6  ),         # accepted difference to EventSetup in T
    OuterRadius  = cms.double( 112.7    ),         # outer radius of outer tracker in cm
    InnerRadius  = cms.double(  21.8    ),         # inner radius of outer tracker in cm
    MaxPitch     = cms.double(    .01   )          # max strip/pixel pitch of outer tracker sensors in cm
  )

)
