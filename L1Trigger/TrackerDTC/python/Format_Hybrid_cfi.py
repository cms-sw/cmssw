import FWCore.ParameterSet.Config as cms

TrackerDTCFormat_params = cms.PSet (

  #=== specific format parameter

  ParamsTTStubAlgo = cms.PSet (

    CheckHistory   = cms.bool  ( False ),                                         # check consitency between configured TTStub algo and the one used during input sample production
    Label          = cms.string( "TTStubAlgorithm_official_Phase2TrackerDigi_" ), # producer name used during input sample production
    Process        = cms.string( "HLT" ),                                         # process name used during input sample production
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

    LayerRs      = cms.vdouble(  24.9316,  37.1777,  52.2656,  68.7598,  86.0156, 108.3105 ), # mean radius of outer tracker barrel layer
    DiskZs       = cms.vdouble( 131.1914, 154.9805, 185.3320, 221.6016, 265.0195           ), # mean z of outer tracker endcap disks
    Disk2SRsSet  = cms.VPSet(                                                                 # center radius of outer tracker endcap 2S diks strips
      cms.PSet( Disk2SRs = cms.vdouble( 66.4391, 71.4391, 76.2750, 81.2750, 82.9550, 87.9550, 93.8150, 98.8150, 99.8160, 104.8160 ) ), # disk 1
      cms.PSet( Disk2SRs = cms.vdouble( 66.4391, 71.4391, 76.2750, 81.2750, 82.9550, 87.9550, 93.8150, 98.8150, 99.8160, 104.8160 ) ), # disk 2
      cms.PSet( Disk2SRs = cms.vdouble( 63.9903, 68.9903, 74.2750, 79.2750, 81.9562, 86.9562, 92.4920, 97.4920, 99.8160, 104.8160 ) ), # disk 3
      cms.PSet( Disk2SRs = cms.vdouble( 63.9903, 68.9903, 74.2750, 79.2750, 81.9562, 86.9562, 92.4920, 97.4920, 99.8160, 104.8160 ) ), # disk 4
      cms.PSet( Disk2SRs = cms.vdouble( 63.9903, 68.9903, 74.2750, 79.2750, 81.9562, 86.9562, 92.4920, 97.4920, 99.8160, 104.8160 ) )  # disk 5
    )

  )

)