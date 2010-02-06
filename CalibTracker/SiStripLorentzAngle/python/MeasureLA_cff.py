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
    return cms.PSet( Method = cms.int32(method),
                     Granularity = granularity,
                     ReportName = cms.string(filename) )

def LA_Measurement(method, granularity, minimumEntries, maxChi2ndof) :
    return cms.PSet( Method = cms.int32(method),
                     Granularity = granularity,
                     MinEntries = cms.uint32(minimumEntries),
                     MaxChi2ndof = cms.double(maxChi2ndof) )

def LA_Calibration(method, pitch, slope, offset, pull) :
    return cms.PSet( Method = cms.int32(method),
                     Pitch = cms.uint32(pitch),
                     Slope = cms.double(slope),
                     Offset = cms.double(offset),
                     ErrorScaling = cms.double(pull) )

LorentzAngleCalibrations_PeakModeBEAM = cms.VPSet(
    LA_Calibration( METHOD_MULTI , 183, 0.993787,  0.000339113, 1.15299),
    LA_Calibration( METHOD_MULTI , 122, 0.997,     0.000135394, 1.14415),
    LA_Calibration( METHOD_MULTI , 120, 0.981605, -0.000719444, 1.19),
    LA_Calibration( METHOD_MULTI ,  80, 0.969566, -0.00249146,  1.4683),
    LA_Calibration( METHOD_SYMM  , 183, 1.02293,   0.000835307, 0.954484),
    LA_Calibration( METHOD_SYMM  , 122, 1.00257,   0.000463179, 1.0088),
    LA_Calibration( METHOD_SYMM  , 120, 1.01526,   0.00654113,  0.962509),
    LA_Calibration( METHOD_SYMM  ,  80, 1.02857,   0.00717154,  1.06112),
    LA_Calibration( METHOD_SQRTVAR,183, 1.06389,  -0.00019858,  1.20346),
    LA_Calibration( METHOD_SQRTVAR,122, 1.01179,   0.000105374, 1.19306),
    LA_Calibration( METHOD_SQRTVAR,120, 1.01375,   0.00573414,  1.25892),
    LA_Calibration( METHOD_SQRTVAR, 80, 1.00877,   0.00469918,  1.27583),
    LA_Calibration( METHOD_RATIO , 183, 0.975289, -0.000382602, 0.70125),
    LA_Calibration( METHOD_RATIO , 122, 0.996539, -0.0010189,   0.739489),
    LA_Calibration( METHOD_RATIO , 120, 0.997464,  0.00207606,  0.490908),
    LA_Calibration( METHOD_RATIO ,  80, 0.994562,  0.000217672, 0.573467),
    LA_Calibration( METHOD_WIDTH , 183, 1.00262,   0.00126162,  1.15666),
    LA_Calibration( METHOD_WIDTH , 122, 1.01914,   0.00234315,  1.15645),
    LA_Calibration( METHOD_WIDTH , 120, 1.00513,   0.00533482,  1.14288),
    LA_Calibration( METHOD_WIDTH ,  80, 0.994517,  0.0029991,   1.28047)
    )


LorentzAngleCalibrations_DeconvolutionModeBEAM = cms.VPSet(
    LA_Calibration( METHOD_MULTI , 183, 1.01471,  -0.000323381, 1.14294),
    LA_Calibration( METHOD_MULTI , 122, 1.00489,  -0.00036418,  1.14714),
    LA_Calibration( METHOD_MULTI , 120, 1.01945,  -0.00198754,  1.29889),
    LA_Calibration( METHOD_MULTI ,  80, 1.02398,   0.0021505,   1.41295),
    LA_Calibration( METHOD_SYMM  , 183, 1.03743,   0.00405498,  1.0072),
    LA_Calibration( METHOD_SYMM  , 122, 0.989113,  0.00126216,  0.972386),
    LA_Calibration( METHOD_SYMM  , 120, 1.03347,   0.032091,    1.07075),
    LA_Calibration( METHOD_SYMM  ,  80, 1.03156,   0.0150397,   1.03983),
    LA_Calibration( METHOD_SQRTVAR,183, 1.2466,   -0.000791538, 1.45773),
    LA_Calibration( METHOD_SQRTVAR,122, 1.0839,    0.00403472,  1.26539),
    LA_Calibration( METHOD_SQRTVAR,120, 0.793985,  0.0143724,   1.26734),
    LA_Calibration( METHOD_SQRTVAR, 80, 1.00799,   0.0140537,   1.36507),
    LA_Calibration( METHOD_RATIO , 183, 0.8904,   -0.00166737,  1.00074),
    LA_Calibration( METHOD_RATIO , 122, 0.479319, -0.0899813,   0.652796),
    LA_Calibration( METHOD_RATIO , 120, 0.745029,  0.01053,     1.03559),
    LA_Calibration( METHOD_RATIO ,  80, 0.897227,  0.00171487,  1.03397),
    LA_Calibration( METHOD_WIDTH, 183,  1.70345,   0.0374983,   1.633),
    LA_Calibration( METHOD_WIDTH, 122,  1.45506,   0.0407221,   1.4022),
    LA_Calibration( METHOD_WIDTH, 120, -0.116306,  0.0136894,   2.82387),
    LA_Calibration( METHOD_WIDTH,  80,  0.852886,  0.012205,    1.33618)
    )

LorentzAngleCalibrations_PeakModeCOSMIC = cms.VPSet(
    LA_Calibration( METHOD_MULTI , 183, 1.0,  0.0, 1.15),
    LA_Calibration( METHOD_MULTI , 122, 1.0,  0.0, 1.15),
    LA_Calibration( METHOD_MULTI , 120, 1.0,  0.0, 1.15),
    LA_Calibration( METHOD_MULTI ,  80, 1.0,  0.0, 1.15)
    )


LorentzAngleCalibrations_DeconvolutionModeCOSMIC = cms.VPSet(
    LA_Calibration( METHOD_MULTI , 183, 1.0,  0.0, 1.15),
    LA_Calibration( METHOD_MULTI , 122, 1.0,  0.0, 1.15),
    LA_Calibration( METHOD_MULTI , 120, 1.0,  0.0, 1.15),
    LA_Calibration( METHOD_MULTI ,  80, 1.0,  0.0, 1.15)
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
