import FWCore.ParameterSet.Config as cms

byLayer = cms.bool(False)
byModule = cms.bool(True)
METHOD_WIDTH = cms.int32(1)
METHOD_RATIO = cms.int32(2)
METHOD_SQRTVAR = cms.int32(4)
METHOD_SYMM = cms.int32(8)

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
    LA_Calibration( METHOD_SYMM, 183, 1.02351,  0.000723037, 0.921798),
    LA_Calibration( METHOD_SYMM, 122, 0.999362, 0.000454755, 0.913588),
    LA_Calibration( METHOD_SYMM, 120, 1.02851,  0.00838982,  0.969561),
    LA_Calibration( METHOD_SYMM,  80, 1.04731,  0.00955597,  0.982278),
    LA_Calibration( METHOD_SQRTVAR, 183, 1.06516, -0.000143231, 1.21116),
    LA_Calibration( METHOD_SQRTVAR, 122, 1.01051, -7.32838e-05, 1.19388),
    LA_Calibration( METHOD_SQRTVAR, 120, 1.01642,  0.00585199,  1.19422),
    LA_Calibration( METHOD_SQRTVAR,  80, 1.0146,   0.00486891,  1.28626),
    LA_Calibration( METHOD_RATIO,   183, 0.981047, 0.000140891, 0.0354449),
    LA_Calibration( METHOD_RATIO,   122, 1.00273, -0.00136105,  0.0770203),
    LA_Calibration( METHOD_RATIO,   120, 1.00007,  0.00206247,  0.0390019),
    LA_Calibration( METHOD_RATIO,    80, 0.996719, 0.000287906, 0.0259316),
    LA_Calibration( METHOD_WIDTH,   183, 1.00847,  0.00147708,  1.18568),
    LA_Calibration( METHOD_WIDTH,   122, 1.01785,  0.00219077,  1.12721),
    LA_Calibration( METHOD_WIDTH,   120, 1.00736,  0.00551022,  1.14293),
    LA_Calibration( METHOD_WIDTH,    80, 0.997497, 0.00296953,  1.24678)
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
