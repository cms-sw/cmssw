import FWCore.ParameterSet.Config as cms

TrackerDTCFormat_params = cms.PSet (

  #=== specific format parameter

  ParamsFormat = cms.PSet (

    MaxEta         = cms.double(   2.4  ), # cut on stub eta
    MinPt          = cms.double(   3.   ), # cut on stub pt, also defines region overlap shape in GeV
    ChosenRofPhi   = cms.double(  67.24 ), # critical radius defining region overlap shape in cm

    WidthR         = cms.int32 (  12    ), # number of bits used for stub r - ChosenRofPhi
    WidthPhi       = cms.int32 (  15    ), # number of bits used for stub phi w.r.t. region centre
    WidthZ         = cms.int32 (  14    ), # number of bits used for stub z
    NumLayers      = cms.int32 (   7    ), # number of detector layers a reconstructbale particle may cross
    NumSectorsPhi  = cms.int32 (   2    ), # number of phi sectors used in hough transform
    NumBinsQoverPt = cms.int32 (  16    ), # number of qOverPt bins used in hough transform
    NumBinsPhiT    = cms.int32 (  32    ), # number of phiT bins used in hough transform

    ChosenRofZ     = cms.double(  50.   ), # critical radius defining r-z sector shape in cm
    BeamWindowZ    = cms.double(  15.   ), # half lumi region size in cm
    HalfLength     = cms.double( 270.   ), # has to be >= max stub z / 2 in cm
    BoundariesEta  = cms.vdouble( -2.40, -2.08, -1.68, -1.26, -0.90, -0.62, -0.41, -0.20, 0.0, 0.20, 0.41, 0.62, 0.90, 1.26, 1.68, 2.08, 2.40 ) # defining r-z sector shape

  )

)