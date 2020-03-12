import FWCore.ParameterSet.Config as cms

TrackerDTCFormat_params = cms.PSet (

  #=== specific format parameter

  ParamsTTStubAlgo = cms.PSet (

    Label          = cms.string( "TTStubAlgorithm_official_Phase2TrackerDigi_" ), # 
    Process        = cms.string( "HLT" ),                                         # empty string possible
    BaseWindowSize = cms.double( .5 )                                             # precision of window sizes in pitch units

  ),

  #=== specific format parameter

  ParamsFormat = cms.PSet (

    MaxEta       = cms.double(  2.5 ), # cut on stub eta
    MinPt        = cms.double(  2.  ), # cut on stub pt, also defines region overlap shape in GeV
    ChosenRofPhi = cms.double( 55.  ), # critical radius defining region overlap shape in cm
    NumLayers    = cms.int32 (  4   ), # max number of detector layer connected to one DTC
    NumRingsPS   = cms.vint32( 11, 11, 8, 8, 8 ), # number of outer PS rings for disk 1, 2, 3, 4, 5

    WidthsR      = cms.vint32 (   7,     7,    12,      7   ), # number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsZ      = cms.vint32 (  12,     8,     7,      7   ), # number of bits used for stub z w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsPhi    = cms.vint32 (  14,    17,    14,     14   ), # number of bits used for stub phi w.r.t. region centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsAlpha  = cms.vint32 (   0,     0,     0,      4   ), # number of bits used for stub row number for module types (barrelPS, barrel2S, diskPS, disk2S)
    WidthsBend   = cms.vint32 (   3,     4,     3,      4   ), # number of bits used for stub bend number for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesR      = cms.vdouble(   7.5,   7.5, 120. ,    0.  ), # range in stub r which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesZ      = cms.vdouble( 240.,  240.,    7.5,    7.5 ), # range in stub z which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    RangesAlpha  = cms.vdouble(   0.,    0.,    0.,  1024.  ), # range in stub row which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)

    LayerRs      = cms.vdouble(  24.8656,  37.1678,  52.2700,  68.7000,  86.0000, 110.8000 ), # mean radius of outer tracker barrel layer
    DiskZs       = cms.vdouble( 131.1800, 155.0000, 185.3400, 221.6190, 265.0000           ), # mean z of outer tracker endcap disks
    Disk2SRsSet  = cms.VPSet(                                                                 # center radius of outer tracker endcap 2S diks strips
      cms.PSet( Disk2SRs = cms.vdouble( 66.7345, 71.7345, 77.5056, 82.5056, 84.8444, 89.8444, 95.7515, 100.7515, 102.475, 107.475 ) ), # disk 1
      cms.PSet( Disk2SRs = cms.vdouble( 66.7345, 71.7345, 77.5056, 82.5056, 84.8444, 89.8444, 95.7515, 100.7515, 102.475, 107.475 ) ), # disk 2
      cms.PSet( Disk2SRs = cms.vdouble( 65.1317, 70.1317, 75.6300, 80.6300, 83.9293, 88.9293, 94.6316,  99.6316, 102.475, 107.475 ) ), # disk 3
      cms.PSet( Disk2SRs = cms.vdouble( 65.1317, 70.1317, 75.6300, 80.6300, 83.9293, 88.9293, 94.6316,  99.6316, 102.475, 107.475 ) ), # disk 4
      cms.PSet( Disk2SRs = cms.vdouble( 65.1317, 70.1317, 75.6300, 80.6300, 83.9293, 88.9293, 94.6316,  99.6316, 102.475, 107.475 ) )  # disk 5
    )

  )

)