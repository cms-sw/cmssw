import FWCore.ParameterSet.Config as cms

byLayer = cms.int32(0)
byModule = cms.int32(1)
byModuleSummary = cms.int32(2)

METHOD_WIDTH =   1  
METHOD_RATIO =   2  
METHOD_SQRTVAR = 4
METHOD_SYMM =    8 
METHOD_MULTI =  16

def LA_Report(method, granularity, filename) :
    return cms.PSet( Method = method,
                     Granularity = granularity,
                     ReportName = cms.string(filename) )

def LA_Measurement(method, granularity, minimumEntries, maxChi2ndof) :
    return cms.PSet( Method = method,
                     Granularity = granularity,
                     MinEntries = cms.uint32(minimumEntries),
                     MaxChi2ndof = cms.double(maxChi2ndof) )

def LA_Calibration(method, pitch, slope, offset, pull) :
    return cms.PSet( Method = cms.int32(method),
                     Pitch = cms.uint32(pitch),
                     Slope = cms.double(slope),
                     Offset = cms.double(offset),
                     ErrorScaling = cms.double(pull) )

LorentzAngleCalibrations_PeakMode = cms.VPSet(
    LA_Calibration( METHOD_MULTI, 183, 1, 0, 1),
    LA_Calibration( METHOD_MULTI, 122, 1, 0, 1),
    LA_Calibration( METHOD_MULTI, 120, 1, 0, 1),
    LA_Calibration( METHOD_MULTI,  80, 1, 0, 1),
    LA_Calibration( METHOD_SYMM, 183, 1.02556, 0.000940537 , 0.905345),
    LA_Calibration( METHOD_SYMM, 122, 1.00056, 0.000486183,  0.971576),
    LA_Calibration( METHOD_SYMM, 120, 1.04376, 0.0102358,    0.928914),
    LA_Calibration( METHOD_SYMM,  80, 1.06168, 0.012654 ,    0.956278),
    LA_Calibration( METHOD_SQRTVAR, 183, 1.06409, -0.000177808, 1.2062),
    LA_Calibration( METHOD_SQRTVAR, 122, 1.01262, 0.000167946, 1.18959),
    LA_Calibration( METHOD_SQRTVAR, 120, 1.01341, 0.00572721 , 1.2561 ),
    LA_Calibration( METHOD_SQRTVAR,  80, 1.0091, 0.00471296  , 1.27687),
    LA_Calibration( METHOD_RATIO,   183, 0.974597, -0.000213534, 0.0346719),
    LA_Calibration( METHOD_RATIO,   122, 1.00448, -0.000735487, 0.0792406 ),
    LA_Calibration( METHOD_RATIO,   120, 0.997966, 0.00187802 , 0.0388548 ),
    LA_Calibration( METHOD_RATIO,    80, 0.994399, 0.000230179, 0.0253525),
    LA_Calibration( METHOD_WIDTH,   183, 1.00262, 0.00126162 , 1.15666 ),
    LA_Calibration( METHOD_WIDTH,   122, 1.01914, 0.00234315 , 1.15645 ),
    LA_Calibration( METHOD_WIDTH,   120, 1.00513, 0.00533482 , 1.14288),
    LA_Calibration( METHOD_WIDTH,    80, 0.994517, 0.0029991, 1.28047 )
    )


LorentzAngleCalibrations_DeconvolutionMode = cms.VPSet(
    LA_Calibration( METHOD_MULTI, 183, 1, 0, 1),
    LA_Calibration( METHOD_MULTI, 122, 1, 0, 1),
    LA_Calibration( METHOD_MULTI, 120, 1, 0, 1),
    LA_Calibration( METHOD_MULTI,  80, 1, 0, 1),
    LA_Calibration( METHOD_SYMM, 183, 1.03812, 0.00424726 , 0.957549 ),
    LA_Calibration( METHOD_SYMM, 122, 0.990598, 0.00138539, 0.952638),
    LA_Calibration( METHOD_SYMM, 120, 1.05111, 0.0347653 , 0.946016 ),
    LA_Calibration( METHOD_SYMM,  80, 1.05226, 0.0208008 , 0.928397  ),
    LA_Calibration( METHOD_SQRTVAR, 183, 1.24622, -0.000833614, 1.4618),
    LA_Calibration( METHOD_SQRTVAR, 122, 1.08341, 0.00398573, 1.25087),
    LA_Calibration( METHOD_SQRTVAR, 120, 0.794449, 0.0143725 , 1.27355  ),
    LA_Calibration( METHOD_SQRTVAR,  80, 1.00867,  0.0140923 , 1.36218 ),
    LA_Calibration( METHOD_RATIO, 183,  1,0,1),
    LA_Calibration( METHOD_RATIO, 122,  1,0,1),
    LA_Calibration( METHOD_RATIO, 120,  1,0,1),
    LA_Calibration( METHOD_RATIO,  80,  1,0,1),
    LA_Calibration( METHOD_WIDTH, 183,  1,0,1),
    LA_Calibration( METHOD_WIDTH, 122,  1,0,1),
    LA_Calibration( METHOD_WIDTH, 120,  1,0,1),
    LA_Calibration( METHOD_WIDTH,  80,  1,0,1)
    )

LorentzAngleCalibrations_Uncalibrated = cms.VPSet(
    LA_Calibration( METHOD_MULTI, 183, 1, 0, 1),
    LA_Calibration( METHOD_MULTI, 122, 1, 0, 1),
    LA_Calibration( METHOD_MULTI, 120, 1, 0, 1),
    LA_Calibration( METHOD_MULTI,  80, 1, 0, 1),
    LA_Calibration( METHOD_SYMM, 183, 1, 0, 1),
    LA_Calibration( METHOD_SYMM, 122, 1, 0, 1),
    LA_Calibration( METHOD_SYMM, 120, 1, 0, 1),
    LA_Calibration( METHOD_SYMM,  80, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR, 183, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR, 122, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR, 120, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR,  80, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   183, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   122, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   120, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,    80, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   183, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   122, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   120, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,    80, 1, 0, 1)
    )
