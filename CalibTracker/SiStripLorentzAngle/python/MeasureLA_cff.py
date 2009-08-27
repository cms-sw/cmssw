import FWCore.ParameterSet.Config as cms

byLayer = cms.bool(False)
byModule = cms.bool(True)
METHOD_WIDTH = cms.int32(1)
METHOD_RATIO = cms.int32(2)
METHOD_SQRTVAR = cms.int32(4)

def LA_Report(method, granularity, filename) :
    return cms.PSet( Method = method,
                     ByModule = granularity,
                     ReportName = cms.string(filename) )

def LA_Measurement(method, granularity, minimumEntries, maxChi2ndof) :
    return cms.PSet( Method = method,
                     ByModule = granularity,
                     MinEntries = cms.uint32(minimumEntries),
                     MaxChi2ndof = cms.double(maxChi2ndof) )

def LA_Calibration(method, pitch, slope, offset, pull) :
    return cms.PSet( Method = method,
                     Pitch = cms.uint32(pitch),
                     Slope = cms.double(slope),
                     Offset = cms.double(offset),
                     ErrorScaling = cms.double(pull) )


LorentzAngleCalibrations_PeakMode = cms.VPSet(
    LA_Calibration( METHOD_SQRTVAR, 183, 1.0289, -0.000320368, 0.876556),
    LA_Calibration( METHOD_SQRTVAR, 122, 0.992503, -0.000829539, 0.755962),
    LA_Calibration( METHOD_SQRTVAR, 120, 1.01791, 0.00587959, 0.745295),
    LA_Calibration( METHOD_SQRTVAR,  80, 1.00784, 0.00277018,  0.874995),
    LA_Calibration( METHOD_RATIO,   183, 0.97007,  -0.00107811, 10.3975),
    LA_Calibration( METHOD_RATIO,   122, 0.944102, -0.00706026,  3.76948),
    LA_Calibration( METHOD_RATIO,   120, 1.00205,   0.00255656,   9.81462),
    LA_Calibration( METHOD_RATIO,    80, 0.996139,  0.000359434, 18.1353),
    LA_Calibration( METHOD_WIDTH,   183, 0.991679, 0.000813695, 0.884903),
    LA_Calibration( METHOD_WIDTH,   122, 1.0285, 0.00322536, 0.779108),
    LA_Calibration( METHOD_WIDTH,   120, 1.01354, 0.00648133, 0.821002),
    LA_Calibration( METHOD_WIDTH,    80, 1.01182, 0.00450226, 0.843897)
    )

LorentzAngleCalibrations_DeconvolutionMode = cms.VPSet(
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
